/*
 * llama.c - Pure C inference engine for Llama 3.2-1B
 *
 * Supports:
 *   - SafeTensors format parsing (F16 / BF16 / F32)
 *   - RMSNorm, RoPE, GQA (Grouped Query Attention)
 *   - SwiGLU feed-forward network
 *   - Greedy & temperature sampling
 *   - Switchable BLAS backend: OpenBLAS (default) or Intel MKL
 *
 * ─────────────────────── Build Instructions ────────────────────────────
 *
 *  OpenBLAS (default):
 *    gcc -O3 -march=native -o llama llama.c -lm -lopenblas
 *
 *  Intel MKL:
 *    gcc -O3 -march=native -DUSE_MKL -o llama_mkl llama.c -lm -lmkl_rt
 *
 *  With explicit include paths (if needed):
 *    # OpenBLAS
 *    gcc -O3 -march=native -o llama llama.c -lm -lopenblas \
 *        -I/usr/include/x86_64-linux-gnu
 *    # MKL
 *    gcc -O3 -march=native -DUSE_MKL -o llama_mkl llama.c -lm -lmkl_rt \
 *        -I/usr/include/mkl
 *
 *  Quick benchmark (both backends, same prompt):
 *    ./bench.sh <model_dir> "<prompt>" [max_tokens]
 *    (bench.sh is generated alongside this file)
 *
 * ────────────────────────── Usage ──────────────────────────────────────
 *
 *  ./llama     <model_dir> "<prompt>" [max_tokens] [temperature]
 *  ./llama_mkl <model_dir> "<prompt>" [max_tokens] [temperature]
 *
 *  model_dir should contain:
 *    model.safetensors  (or model-00001-of-00002.safetensors etc.)
 *    tokenizer.json
 *
 * ───────────────────────── Backend Notes ───────────────────────────────
 *
 *  Both backends expose an identical cblas_sgemv interface.
 *  MKL is Intel's proprietary library, highly tuned for Intel CPUs:
 *    - Dispatches at runtime to the best AVX/AVX2/AVX-512 kernel
 *    - Better cache blocking on Intel microarchitectures
 *    - Typically 10-40% faster than OpenBLAS on Intel hardware
 *    - On AMD CPUs, OpenBLAS often wins or ties
 *  OpenBLAS is open-source and portable; performs well on both x86 and ARM.
 *
 *  BLAS_BACKEND env var (runtime switch for the MKL binary):
 *    MKL_VERBOSE=1 ./llama_mkl ...    # show MKL dispatch decisions
 *    MKL_NUM_THREADS=1 ./llama_mkl ... # force single-threaded MKL
 */

/* Required for clock_gettime, strdup, and other POSIX extensions */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/* ─────────────────────────── BLAS Backend ──────────────────────────────
 *
 *  Compile with -DUSE_MKL  → Intel MKL  (link: -lmkl_rt)
 *  Default (no flag)       → OpenBLAS   (link: -lopenblas)
 *
 *  Both provide identical cblas_sgemv, so all math code below is
 *  100% shared — only the header and the runtime banner differ.
 */
#ifdef USE_MKL
  #include <mkl_cblas.h>
  #define BLAS_BACKEND_NAME "Intel MKL"
#else
  #include <cblas.h>
  #define BLAS_BACKEND_NAME "OpenBLAS"
#endif

/* ─────────────────────────── model config ──────────────────────────── */

typedef struct {
    int   dim;           /* hidden size                  (2048) */
    int   n_layers;      /* number of transformer layers (16)   */
    int   n_heads;       /* query heads                  (32)   */
    int   n_kv_heads;    /* key/value heads (GQA)        (8)    */
    int   ff_dim;        /* intermediate size            (8192) */
    int   vocab_size;    /* vocabulary size              (128256)*/
    int   max_seq_len;   /* max context length           (131072)*/
    float rope_theta;    /* RoPE base                    (500000)*/
    int   head_dim;      /* dim / n_heads                        */
    int   gqa_factor;    /* n_heads / n_kv_heads                 */
} Config;

/* Llama 3.2-1B defaults */
static Config llama32_1b_config(void) {
    Config c;
    c.dim         = 2048;
    c.n_layers    = 16;
    c.n_heads     = 32;
    c.n_kv_heads  = 8;
    c.ff_dim      = 8192;
    c.vocab_size  = 128256;
    c.max_seq_len = 2048;   /* reduce for RAM; model supports 131072 */
    c.rope_theta  = 500000.0f;
    c.head_dim    = c.dim / c.n_heads;
    c.gqa_factor  = c.n_heads / c.n_kv_heads;
    return c;
}

/* ─────────────────────────── safetensors ───────────────────────────── */

typedef enum { ST_F32 = 0, ST_BF16 = 1, ST_I32 = 2, ST_F16 = 3, ST_UNKNOWN } STDtype;

typedef struct {
    char     name[256];
    STDtype  dtype;
    uint64_t offset_start;
    uint64_t offset_end;
    int      shape[8];
    int      ndim;
} STTensor;

typedef struct {
    int       n;
    STTensor *tensors;
    uint8_t  *data;
    uint64_t  data_len;
    void     *mmap_ptr;
    size_t    mmap_len;
    int       fd;
} SafeTensors;

static int json_skip_ws(const char *s, int i) {
    while (s[i]==' '||s[i]=='\n'||s[i]=='\r'||s[i]=='\t') i++;
    return i;
}
static int json_read_string(const char *s, int i, char *out, int max) {
    if (s[i] != '"') return -1;
    i++;
    int j = 0;
    while (s[i] && s[i] != '"') {
        if (s[i]=='\\') i++;
        if (j < max-1) out[j++] = s[i];
        i++;
    }
    out[j] = 0;
    return s[i]=='"' ? i+1 : -1;
}
static int json_read_uint64(const char *s, int i, uint64_t *v) {
    *v = 0;
    if (s[i]<'0'||s[i]>'9') return -1;
    while (s[i]>='0'&&s[i]<='9') { *v = *v*10 + (s[i]-'0'); i++; }
    return i;
}

static int parse_tensor_entry(const char *hdr, int i, STTensor *t) {
    i = json_skip_ws(hdr, i);
    if (hdr[i] != '{') return -1; i++;

    memset(t, 0, sizeof(*t));
    t->dtype = ST_UNKNOWN;

    int found_dtype=0, found_data=0;
    for (;;) {
        i = json_skip_ws(hdr, i);
        if (hdr[i] == '}') { i++; break; }
        if (hdr[i] == ',') { i++; continue; }

        char key[64];
        i = json_read_string(hdr, i, key, sizeof(key));
        if (i<0) return -1;
        i = json_skip_ws(hdr, i);
        if (hdr[i]!=':') return -1; i++;
        i = json_skip_ws(hdr, i);

        if (strcmp(key,"dtype")==0) {
            char dstr[32];
            i = json_read_string(hdr, i, dstr, sizeof(dstr));
            if (i<0) return -1;
            if      (strcmp(dstr,"F32" )==0) t->dtype = ST_F32;
            else if (strcmp(dstr,"BF16")==0) t->dtype = ST_BF16;
            else if (strcmp(dstr,"F16" )==0) t->dtype = ST_F16;
            else if (strcmp(dstr,"I32" )==0) t->dtype = ST_I32;
            found_dtype = 1;
        } else if (strcmp(key,"shape")==0) {
            if (hdr[i]!='[') return -1; i++;
            while (hdr[i]!=']') {
                i = json_skip_ws(hdr, i);
                if (hdr[i]==',') { i++; continue; }
                uint64_t v; i = json_read_uint64(hdr, i, &v);
                if (i<0) return -1;
                if (t->ndim < 8) t->shape[t->ndim++] = (int)v;
            }
            i++;
        } else if (strcmp(key,"data_offsets")==0) {
            if (hdr[i]!='[') return -1; i++;
            i = json_skip_ws(hdr, i);
            i = json_read_uint64(hdr, i, &t->offset_start); if (i<0) return -1;
            i = json_skip_ws(hdr, i);
            if (hdr[i]==',') i++;
            i = json_skip_ws(hdr, i);
            i = json_read_uint64(hdr, i, &t->offset_end);   if (i<0) return -1;
            i = json_skip_ws(hdr, i);
            if (hdr[i]==']') i++;
            found_data = 1;
        } else {
            if (hdr[i]=='"') {
                char tmp[256]; i = json_read_string(hdr, i, tmp, sizeof(tmp));
            } else if (hdr[i]>='0'&&hdr[i]<='9') {
                uint64_t v; i = json_read_uint64(hdr, i, &v);
            } else {
                while (hdr[i]&&hdr[i]!=','&&hdr[i]!='}') i++;
            }
        }
    }
    (void)found_dtype; (void)found_data;
    return i;
}

static int st_open(SafeTensors *st, const char *path) {
    st->fd = open(path, O_RDONLY);
    if (st->fd < 0) { perror(path); return -1; }

    struct stat sb;
    if (fstat(st->fd, &sb) < 0) { perror("fstat"); return -1; }
    st->mmap_len = (size_t)sb.st_size;

    st->mmap_ptr = mmap(NULL, st->mmap_len, PROT_READ, MAP_PRIVATE, st->fd, 0);
    if (st->mmap_ptr == MAP_FAILED) { perror("mmap"); return -1; }

    uint8_t *raw = (uint8_t*)st->mmap_ptr;

    uint64_t hdr_len = 0;
    for (int b=0; b<8; b++) hdr_len |= ((uint64_t)raw[b]) << (8*b);

    char *hdr    = (char*)raw + 8;
    st->data     = raw + 8 + hdr_len;
    st->data_len = st->mmap_len - 8 - hdr_len;

    int cap = 512;
    st->tensors = (STTensor*)malloc(cap * sizeof(STTensor));
    st->n = 0;

    int i = 0;
    i = json_skip_ws(hdr, i);
    if (hdr[i]!='{') { fprintf(stderr,"Bad header\n"); return -1; } i++;

    for (;;) {
        i = json_skip_ws(hdr, i);
        if (hdr[i]=='}') break;
        if (hdr[i]==',') { i++; continue; }

        char tname[256];
        int ni = json_read_string(hdr, i, tname, sizeof(tname));
        if (ni<0) break;
        i = ni;
        i = json_skip_ws(hdr, i);
        if (hdr[i]!=':') break; i++;
        i = json_skip_ws(hdr, i);

        if (strcmp(tname,"__metadata__")==0) {
            if (hdr[i]=='{') {
                int depth=1; i++;
                while (hdr[i] && depth>0) {
                    if (hdr[i]=='{') depth++;
                    else if (hdr[i]=='}') depth--;
                    i++;
                }
            }
            continue;
        }

        if (st->n >= cap) {
            cap *= 2;
            st->tensors = (STTensor*)realloc(st->tensors, cap*sizeof(STTensor));
        }

        STTensor *t = &st->tensors[st->n];
        int ni2 = parse_tensor_entry(hdr, i, t);
        if (ni2 < 0) { fprintf(stderr,"Failed to parse tensor: %s\n", tname); break; }
        strncpy(t->name, tname, sizeof(t->name)-1);
        t->name[sizeof(t->name)-1] = 0;
        i = ni2;
        st->n++;
    }

    printf("[safetensors] Loaded %d tensors from %s\n", st->n, path);
    return 0;
}

static void st_close(SafeTensors *st) {
    if (st->mmap_ptr && st->mmap_ptr != MAP_FAILED)
        munmap(st->mmap_ptr, st->mmap_len);
    if (st->fd >= 0) close(st->fd);
    free(st->tensors);
}

static STTensor* st_find(SafeTensors *st, const char *name) {
    for (int i=0; i<st->n; i++)
        if (strcmp(st->tensors[i].name, name)==0) return &st->tensors[i];
    return NULL;
}

static float bf16_to_f32(uint16_t v) {
    uint32_t u = ((uint32_t)v) << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

static float f16_to_f32(uint16_t v) {
    uint32_t sign     = (v >> 15) & 0x1;
    uint32_t exp      = (v >> 10) & 0x1f;
    uint32_t mantissa = v & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mantissa == 0) {
            u = sign << 31;
        } else {
            exp = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exp--; }
            mantissa &= 0x3ff;
            u = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        u = (sign << 31) | (0xff << 23) | (mantissa << 13);
    } else {
        u = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    float f; memcpy(&f, &u, 4);
    return f;
}

static float* st_get_f32(SafeTensors *st, STTensor *t, float *scratch, int n) {
    uint8_t *ptr = st->data + t->offset_start;
    if (t->dtype == ST_F32) {
        return (float*)ptr;
    } else if (t->dtype == ST_BF16) {
        uint16_t *bf = (uint16_t*)ptr;
        for (int i=0; i<n; i++) scratch[i] = bf16_to_f32(bf[i]);
        return scratch;
    } else if (t->dtype == ST_F16) {
        uint16_t *fp16 = (uint16_t*)ptr;
        for (int i=0; i<n; i++) scratch[i] = f16_to_f32(fp16[i]);
        return scratch;
    } else {
        fprintf(stderr, "Unsupported dtype for tensor %s\n", t->name);
        return NULL;
    }
}

/* ─────────────────────────── model weights ─────────────────────────── */

typedef struct {
    float  *embed_tokens;
    float **attn_q;
    float **attn_k;
    float **attn_v;
    float **attn_o;
    float **ff_gate;
    float **ff_up;
    float **ff_down;
    float **attn_norm;
    float **ff_norm;
    float  *norm;
    float  *lm_head;
    SafeTensors *st;
} Weights;

static float* alloc_copy(SafeTensors *st, const char *name,
                          float **scratch_ptr, int *scratch_sz) {
    STTensor *t = st_find(st, name);
    if (!t) { fprintf(stderr,"Missing: %s\n", name); return NULL; }
    int n = 1;
    for (int d=0; d<t->ndim; d++) n *= t->shape[d];

    float *buf = (float*)malloc((size_t)n * sizeof(float));
    if (!buf) { fprintf(stderr,"OOM for %s\n", name); return NULL; }

    if (t->dtype != ST_F32 && n > *scratch_sz) {
        float *ns = (float*)realloc(*scratch_ptr, (size_t)n * sizeof(float));
        if (!ns) { fprintf(stderr,"OOM scratch for %s\n", name); free(buf); return NULL; }
        *scratch_ptr = ns;
        *scratch_sz  = n;
    }

    float *src = st_get_f32(st, t, *scratch_ptr, n);
    if (!src) {
        fprintf(stderr, "Cannot convert dtype for: %s (dtype=%d)\n", name, t->dtype);
        free(buf); return NULL;
    }
    memcpy(buf, src, (size_t)n * sizeof(float));
    return buf;
}

static int load_weights(Weights *w, Config *cfg, SafeTensors *st) {
    char name[256];
    int   scratch_sz = 0;
    float *scratch   = NULL;

#define COPY(dst, nm) do { \
    (dst) = alloc_copy(st, (nm), &scratch, &scratch_sz); \
    if (!(dst)) { free(scratch); return -1; } \
} while(0)

    COPY(w->embed_tokens, "model.embed_tokens.weight");

    w->attn_q    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_k    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_v    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_o    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_gate   = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_up     = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_down   = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_norm = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_norm   = (float**)calloc(cfg->n_layers, sizeof(float*));

    for (int l=0; l<cfg->n_layers; l++) {
        snprintf(name,sizeof(name),"model.layers.%d.self_attn.q_proj.weight",l);
        COPY(w->attn_q[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.self_attn.k_proj.weight",l);
        COPY(w->attn_k[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.self_attn.v_proj.weight",l);
        COPY(w->attn_v[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.self_attn.o_proj.weight",l);
        COPY(w->attn_o[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.mlp.gate_proj.weight",l);
        COPY(w->ff_gate[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.mlp.up_proj.weight",l);
        COPY(w->ff_up[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.mlp.down_proj.weight",l);
        COPY(w->ff_down[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.input_layernorm.weight",l);
        COPY(w->attn_norm[l], name);
        snprintf(name,sizeof(name),"model.layers.%d.post_attention_layernorm.weight",l);
        COPY(w->ff_norm[l], name);
    }

    COPY(w->norm, "model.norm.weight");

    STTensor *lm = st_find(st, "lm_head.weight");
    if (lm) {
        COPY(w->lm_head, "lm_head.weight");
        printf("[weights] lm_head: separate tensor\n");
    } else {
        int n = cfg->vocab_size * cfg->dim;
        w->lm_head = (float*)malloc(n * sizeof(float));
        if (!w->lm_head) { fprintf(stderr,"OOM lm_head\n"); return -1; }
        memcpy(w->lm_head, w->embed_tokens, n * sizeof(float));
        printf("[weights] lm_head: tied to embed_tokens\n");
    }

    free(scratch);
    printf("[weights] All weights loaded successfully.\n");
    return 0;
#undef COPY
}

/* ─────────────────────────── tokenizer ─────────────────────────────── */

static void build_byte_decoder(uint32_t enc[256], uint8_t dec[1024]) {
    memset(dec, 0xff, 1024);
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) ||
                        (b >= 161 && b <= 172) ||
                        (b >= 174 && b <= 255);
        if (printable) {
            enc[b] = (uint32_t)b;
        } else {
            enc[b] = 0x100 + n;
            if (n < 1024) dec[n] = (uint8_t)b;
            n++;
        }
    }
}

static int decode_gpt2_token(const char *tok_str, uint8_t *out, int max_out,
                              uint8_t byte_dec[1024]) {
    int out_len = 0;
    const uint8_t *p = (const uint8_t*)tok_str;
    while (*p && out_len < max_out) {
        uint32_t cp;
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xe0) == 0xc0) {
            cp = (*p & 0x1f) << 6; p++;
            cp |= (*p & 0x3f); p++;
        } else if ((*p & 0xf0) == 0xe0) {
            cp = (*p & 0x0f) << 12; p++;
            cp |= (*p & 0x3f) << 6; p++;
            cp |= (*p & 0x3f); p++;
        } else {
            p++; continue;
        }

        uint8_t raw_byte;
        if (cp < 0x100) {
            raw_byte = (uint8_t)cp;
        } else if ((cp - 0x100) < 1024 && byte_dec[cp - 0x100] != 0xff) {
            raw_byte = byte_dec[cp - 0x100];
        } else {
            raw_byte = '?';
        }
        out[out_len++] = raw_byte;
    }
    return out_len;
}

typedef struct { int a, b, result; } Merge;

typedef struct {
    char    **vocab;
    int      *vocab_lens;
    int       vocab_size;
    Merge    *merges;
    int       n_merges;
    int       bos, eos;
    char    **enc_vocab;
    uint8_t   byte_dec[1024];
    uint32_t  byte_enc[256];
} Tokenizer;

/* ── hash map: encoded string → token id ── */
#define HASHMAP_CAP (1 << 18)
typedef struct { const char *key; int val; } HMEntry;
static HMEntry *g_hm = NULL;

static uint32_t hm_hash(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (uint8_t)*s++; h *= 16777619u; }
    return h;
}
static void hm_set(const char *key, int val) {
    uint32_t h = hm_hash(key) & (HASHMAP_CAP-1);
    while (g_hm[h].key && strcmp(g_hm[h].key, key)!=0)
        h = (h+1) & (HASHMAP_CAP-1);
    g_hm[h].key = key;
    g_hm[h].val = val;
}
static int hm_get(const char *key, int def) {
    uint32_t h = hm_hash(key) & (HASHMAP_CAP-1);
    while (g_hm[h].key) {
        if (strcmp(g_hm[h].key, key)==0) return g_hm[h].val;
        h = (h+1) & (HASHMAP_CAP-1);
    }
    return def;
}

/* Read a JSON string value, handling escape sequences including \uXXXX */
static int read_json_str(const char **pp, char *out, int max) {
    const char *p = *pp;
    if (*p != '"') return -1;
    p++;
    int j = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case 'n':  if(j<max-1) out[j++]='\n'; break;
                case 't':  if(j<max-1) out[j++]='\t'; break;
                case 'r':  if(j<max-1) out[j++]='\r'; break;
                case '\\': if(j<max-1) out[j++]='\\'; break;
                case '"':  if(j<max-1) out[j++]='"';  break;
                case '/':  if(j<max-1) out[j++]='/';  break;
                case 'u': {
                    char hex[5]={0};
                    for(int h=0;h<4&&p[1];h++) hex[h]=*++p;
                    uint32_t cp = (uint32_t)strtoul(hex,NULL,16);
                    if (cp < 0x80) {
                        if(j<max-1) out[j++]=(char)cp;
                    } else if (cp < 0x800) {
                        if(j<max-2){ out[j++]=0xc0|(cp>>6); out[j++]=0x80|(cp&0x3f); }
                    } else {
                        if(j<max-3){ out[j++]=0xe0|(cp>>12);
                                     out[j++]=0x80|((cp>>6)&0x3f);
                                     out[j++]=0x80|(cp&0x3f); }
                    }
                    break;
                }
                default: if(j<max-1) out[j++]=*p; break;
            }
        } else {
            if(j<max-1) out[j++]=*p;
        }
        p++;
    }
    out[j]=0;
    if (*p=='"') p++;
    *pp = p;
    return j;
}

/* Skip a complete JSON value starting at *p; return pointer past it. */
static const char* skip_json_value(const char *p) {
    while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
    if (*p == '"') {
        p++;
        while (*p && *p != '"') { if (*p=='\\') p++; p++; }
        if (*p=='"') p++;
        return p;
    }
    if (*p=='{' || *p=='[') {
        char open=*p, close=(*p=='{') ? '}' : ']';
        int depth=1; p++;
        while (*p && depth>0) {
            if (*p=='\\') { p+=2; continue; }
            if (*p=='"') {
                p++;
                while (*p && *p!='"') { if (*p=='\\') p++; p++; }
                if (*p=='"') p++;
                continue;
            }
            if (*p==open)  depth++;
            if (*p==close) depth--;
            p++;
        }
        return p;
    }
    while (*p && *p!=',' && *p!='}' && *p!=']' && *p!=' ' && *p!='\n' && *p!='\r') p++;
    return p;
}

static const char* find_merges_array(const char *js) {
    const char *p = strstr(js, "\"model\"");
    if (!p) return NULL;
    p += 7;
    while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
    if (*p!=':') return NULL; p++;
    while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
    if (*p!='{') return NULL; p++;

    while (*p) {
        while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
        if (*p=='}' || !*p) break;
        if (*p==',') { p++; continue; }
        if (*p!='"') { p++; continue; }

        p++;
        char key[64]={0}; int ki=0;
        while (*p && *p!='"') {
            if (*p=='\\') p++;
            if (ki<63) key[ki++]=*p;
            p++;
        }
        if (*p=='"') p++;
        key[ki]=0;

        while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
        if (*p!=':') break; p++;
        while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;

        if (strcmp(key,"merges")==0) {
            if (*p=='[') return p;
            break;
        }

        p = skip_json_value(p);
    }
    return NULL;
}

static int tok_load(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *js = (char*)malloc(sz+1);
    if (!js) { fclose(f); return -1; }
    if (fread(js, 1, sz, f) != (size_t)sz)
        fprintf(stderr,"Warning: short read tokenizer.json\n");
    js[sz]=0;
    fclose(f);

    build_byte_decoder(tok->byte_enc, tok->byte_dec);

    tok->vocab_size = 128256;
    tok->vocab      = (char**)calloc(tok->vocab_size, sizeof(char*));
    tok->vocab_lens = (int*)calloc(tok->vocab_size, sizeof(int));
    tok->enc_vocab  = (char**)calloc(tok->vocab_size, sizeof(char*));
    tok->bos = 128000;
    tok->eos = 128009;

    /* ── 1. Parse added_tokens (special tokens) ── */
    const char *at = strstr(js, "\"added_tokens\"");
    if (at) {
        const char *p = strchr(at, '[');
        if (p) {
            p++;
            while (*p) {
                while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
                if (*p==']') break;
                if (*p==',') { p++; continue; }
                if (*p!='{') { p++; continue; }

                int id = -1;
                char content[256]={0};
                const char *obj_end = p;
                int depth=1; obj_end++;
                while (*obj_end && depth>0) {
                    if (*obj_end=='{') depth++;
                    else if (*obj_end=='}') depth--;
                    obj_end++;
                }
                char obj[4096]; int olen = (int)(obj_end-p);
                if (olen>4095) olen=4095;
                memcpy(obj, p, olen); obj[olen]=0;

                char *idp = strstr(obj,"\"id\"");
                if (idp) { idp+=4; while(*idp==' '||*idp==':') idp++; id=atoi(idp); }
                char *cp2 = strstr(obj,"\"content\"");
                if (cp2) {
                    cp2+=9; while(*cp2==' '||*cp2==':') cp2++;
                    const char *cp3=(const char*)cp2;
                    read_json_str(&cp3, content, sizeof(content));
                }
                if (id>=0 && id<tok->vocab_size && content[0]) {
                    free(tok->vocab[id]);
                    tok->vocab[id]      = strdup(content);
                    tok->vocab_lens[id] = (int)strlen(content);
                    tok->enc_vocab[id]  = strdup(content);
                }
                p = obj_end;
            }
        }
    }

    /* ── 2. Parse vocab from model.vocab ── */
    const char *model_sec = strstr(js, "\"model\"");
    if (!model_sec) { fprintf(stderr,"No 'model' section\n"); free(js); return -1; }
    const char *vocab_sec = strstr(model_sec, "\"vocab\"");
    if (!vocab_sec) { fprintf(stderr,"No vocab in model section\n"); free(js); return -1; }

    const char *vp = strchr(vocab_sec, '{');
    if (!vp) { free(js); return -1; }
    vp++;

    int loaded = 0;
    while (*vp) {
        while (*vp==' '||*vp=='\n'||*vp=='\r'||*vp=='\t') vp++;
        if (*vp=='}') break;
        if (*vp==',') { vp++; continue; }
        if (*vp!='"') { vp++; continue; }

        char enc_tok[512];
        read_json_str(&vp, enc_tok, sizeof(enc_tok));

        while (*vp==' '||*vp=='\n'||*vp=='\r'||*vp=='\t') vp++;
        if (*vp!=':') continue;
        vp++;
        while (*vp==' ') vp++;

        int id=0; int neg=0;
        if (*vp=='-') { neg=1; vp++; }
        if (*vp<'0'||*vp>'9') continue;
        while (*vp>='0'&&*vp<='9') { id=id*10+(*vp-'0'); vp++; }
        if (neg) id=-id;

        if (id>=0 && id<tok->vocab_size && !tok->enc_vocab[id]) {
            uint8_t raw[512];
            int rlen = decode_gpt2_token(enc_tok, raw, sizeof(raw)-1, tok->byte_dec);
            raw[rlen]=0;

            tok->enc_vocab[id]  = strdup(enc_tok);
            tok->vocab[id]      = malloc(rlen+1);
            if (tok->vocab[id]) {
                memcpy(tok->vocab[id], raw, rlen+1);
                tok->vocab_lens[id] = rlen;
            }
            loaded++;
        }
    }
    printf("[tokenizer] vocab entries loaded: %d\n", loaded);

    /* ── 3. Parse merges ── */
    int merge_cap = 300000;
    tok->merges   = (Merge*)malloc(merge_cap * sizeof(Merge));
    tok->n_merges = 0;

    g_hm = (HMEntry*)calloc(HASHMAP_CAP, sizeof(HMEntry));
    for (int i=0; i<tok->vocab_size; i++)
        if (tok->enc_vocab[i]) hm_set(tok->enc_vocab[i], i);

    const char *merges_arr = find_merges_array(js);
    if (!merges_arr) {
        fprintf(stderr,"[tokenizer] WARNING: could not find merges array\n");
    } else {
        const char *mp = merges_arr;
        if (*mp == '[') mp++;

        while (*mp && tok->n_merges < merge_cap) {
            while (*mp==' '||*mp=='\n'||*mp=='\r'||*mp=='\t') mp++;
            if (*mp==']') break;
            if (*mp==',') { mp++; continue; }
            if (*mp!='"') { mp++; continue; }

            char merge_str[256];
            read_json_str(&mp, merge_str, sizeof(merge_str));

            char *space = strchr(merge_str, ' ');
            if (!space) continue;
            *space = 0;
            char *sa = merge_str, *sb = space+1;

            int ia = hm_get(sa, -1);
            int ib = hm_get(sb, -1);
            if (ia < 0 || ib < 0) continue;

            char merged[512];
            snprintf(merged, sizeof(merged), "%s%s", sa, sb);
            int ir = hm_get(merged, -1);
            if (ir < 0) continue;

            tok->merges[tok->n_merges++] = (Merge){ia, ib, ir};
        }
    }
    printf("[tokenizer] merges loaded: %d\n", tok->n_merges);

    /* ── 4. Fill gaps in vocab ── */
    for (int i=0; i<tok->vocab_size; i++) {
        if (!tok->vocab[i]) {
            char tmp[32]; snprintf(tmp,sizeof(tmp),"<unk%d>",i);
            tok->vocab[i]      = strdup(tmp);
            tok->vocab_lens[i] = (int)strlen(tmp);
        }
    }

    printf("[tokenizer] bos=%d eos=%d\n", tok->bos, tok->eos);
    free(js);
    return 0;
}

static int byte_to_gpt2_str(uint8_t byte, uint32_t byte_enc[256], char *out) {
    uint32_t enc = byte_enc[byte];
    int len = 0;
    if (enc < 0x80) {
        out[len++] = (char)enc;
    } else if (enc < 0x800) {
        out[len++] = (char)(0xc0 | (enc >> 6));
        out[len++] = (char)(0x80 | (enc & 0x3f));
    } else {
        out[len++] = (char)(0xe0 | (enc >> 12));
        out[len++] = (char)(0x80 | ((enc >> 6) & 0x3f));
        out[len++] = (char)(0x80 | (enc & 0x3f));
    }
    out[len] = 0;
    return len;
}

static int tok_encode(Tokenizer *tok, const char *text, int *ids, int max_ids) {
    int seq_cap = 65536;
    int *seq = (int*)malloc(seq_cap * sizeof(int));
    if (!seq) return 0;
    int seq_len = 0;

    const uint8_t *p = (const uint8_t*)text;
    while (*p && seq_len < seq_cap - 1) {
        uint32_t cp;
        int cp_bytes;
        if      ((*p & 0x80) == 0x00) { cp = *p;                                                           cp_bytes = 1; }
        else if ((*p & 0xe0) == 0xc0) { cp = ((uint32_t)(p[0]&0x1f)<<6)  |  (p[1]&0x3f);                  cp_bytes = 2; }
        else if ((*p & 0xf0) == 0xe0) { cp = ((uint32_t)(p[0]&0x0f)<<12) | ((uint32_t)(p[1]&0x3f)<<6) | (p[2]&0x3f); cp_bytes = 3; }
        else if ((*p & 0xf8) == 0xf0) { cp = ((uint32_t)(p[0]&0x07)<<18) | ((uint32_t)(p[1]&0x3f)<<12)| ((uint32_t)(p[2]&0x3f)<<6)|(p[3]&0x3f); cp_bytes = 4; }
        else                           { cp = *p; cp_bytes = 1; }

        uint8_t utf8[4]; int nb = 0;
        if      (cp < 0x80)    { utf8[0] = (uint8_t)cp; nb = 1; }
        else if (cp < 0x800)   { utf8[0] = 0xc0|(cp>>6); utf8[1] = 0x80|(cp&0x3f); nb = 2; }
        else if (cp < 0x10000) { utf8[0] = 0xe0|(cp>>12); utf8[1] = 0x80|((cp>>6)&0x3f); utf8[2] = 0x80|(cp&0x3f); nb = 3; }
        else                   { utf8[0] = 0xf0|(cp>>18); utf8[1] = 0x80|((cp>>12)&0x3f); utf8[2] = 0x80|((cp>>6)&0x3f); utf8[3] = 0x80|(cp&0x3f); nb = 4; }

        char gpt2_str[32]; int glen = 0;
        for (int b = 0; b < nb; b++)
            glen += byte_to_gpt2_str(utf8[b], tok->byte_enc, gpt2_str + glen);
        gpt2_str[glen] = 0;

        int id = hm_get(gpt2_str, -1);
        if (id >= 0) {
            seq[seq_len++] = id;
        }

        p += cp_bytes;
    }

    /* Apply BPE merges in order */
    for (int m = 0; m < tok->n_merges && seq_len > 1; m++) {
        int a = tok->merges[m].a;
        int b = tok->merges[m].b;
        int c = tok->merges[m].result;
        int write = 0;
        for (int r = 0; r < seq_len; r++) {
            if (r < seq_len-1 && seq[r]==a && seq[r+1]==b) {
                seq[write++] = c;
                r++;
            } else {
                seq[write++] = seq[r];
            }
        }
        seq_len = write;
    }

    int n = (seq_len < max_ids-1) ? seq_len : max_ids-1;
    memcpy(ids, seq, n * sizeof(int));
    free(seq);
    return n;
}

static void tok_free(Tokenizer *tok) {
    for (int i=0; i<tok->vocab_size; i++) {
        free(tok->vocab[i]);
        free(tok->enc_vocab[i]);
    }
    free(tok->vocab);
    free(tok->vocab_lens);
    free(tok->enc_vocab);
    free(tok->merges);
    free(g_hm);
    g_hm = NULL;
}

/* ─────────────────────────── math helpers ──────────────────────────── */

static void rmsnorm(float *out, const float *x, const float *w, int dim) {
    float ss = 0.0f;
    for (int i=0; i<dim; i++) ss += x[i]*x[i];
    ss = 1.0f / sqrtf(ss/(float)dim + 1e-5f);
    for (int i=0; i<dim; i++) out[i] = w[i] * (ss * x[i]);
}

/*
 * matmul / matmul_acc
 * ───────────────────
 * out[out_dim] = W[out_dim × in_dim] · x[in_dim]  (row-major W)
 *
 * Dispatches to cblas_sgemv from whichever library was compiled in:
 *   - OpenBLAS: open-source, great on AMD & ARM
 *   - Intel MKL: Intel-proprietary, fastest on Intel CPUs (AVX-512 etc.)
 *
 * The function signature is identical between the two libraries, so
 * all math code is 100% shared — only the #include and link flag differ.
 *
 * cblas_sgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, y, incy)
 *   CblasRowMajor + CblasNoTrans + M=out_dim + N=in_dim  →  y = A · x
 *   beta=0 for matmul (overwrite), beta=1 for matmul_acc (accumulate).
 */
static void matmul(float *out, const float *x, const float *W,
                   int out_dim, int in_dim) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_dim, in_dim,
                1.0f, W, in_dim,
                x, 1,
                0.0f, out, 1);
}

static void matmul_acc(float *out, const float *x, const float *W,
                        int out_dim, int in_dim) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_dim, in_dim,
                1.0f, W, in_dim,
                x, 1,
                1.0f, out, 1);
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i=1; i<n; i++) if (x[i]>mx) mx=x[i];
    float s = 0.0f;
    for (int i=0; i<n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i=0; i<n; i++) x[i] /= s;
}

static void silu(float *x, int n) {
    for (int i=0; i<n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
                  int pos, float theta) {
    int half = head_dim / 2;
    for (int h=0; h<n_heads; h++) {
        float *qh = q + h*head_dim;
        for (int i=0; i<half; i++) {
            float freq  = 1.0f / powf(theta, (float)(2*i)/(float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float q0 = qh[i], q1 = qh[i+half];
            qh[i]      = q0*c - q1*s;
            qh[i+half] = q0*s + q1*c;
        }
    }
    for (int h=0; h<n_kv_heads; h++) {
        float *kh = k + h*head_dim;
        for (int i=0; i<half; i++) {
            float freq  = 1.0f / powf(theta, (float)(2*i)/(float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float k0 = kh[i], k1 = kh[i+half];
            kh[i]      = k0*c - k1*s;
            kh[i+half] = k0*s + k1*c;
        }
    }
}

/* ─────────────────────────── run state ─────────────────────────────── */

typedef struct {
    float *x;
    float *xb;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *ff_buf;
    float *ff_buf2;
    float *k_cache;
    float *v_cache;
    int    n_layers;
    int    n_kv_heads;
    int    head_dim;
    int    max_seq;
} RunState;

static int alloc_run_state(RunState *s, Config *cfg) {
    int dim    = cfg->dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;
    int q_dim  = cfg->n_heads    * cfg->head_dim;

    s->n_layers   = cfg->n_layers;
    s->n_kv_heads = cfg->n_kv_heads;
    s->head_dim   = cfg->head_dim;
    s->max_seq    = cfg->max_seq_len;

    s->x       = (float*)calloc(dim,                               sizeof(float));
    s->xb      = (float*)calloc(dim,                               sizeof(float));
    s->q       = (float*)calloc(q_dim,                             sizeof(float));
    s->k       = (float*)calloc(kv_dim,                            sizeof(float));
    s->v       = (float*)calloc(kv_dim,                            sizeof(float));
    s->att     = (float*)calloc(cfg->n_heads * cfg->max_seq_len,   sizeof(float));
    s->logits  = (float*)calloc(cfg->vocab_size,                   sizeof(float));
    s->ff_buf  = (float*)calloc(cfg->ff_dim,                       sizeof(float));
    s->ff_buf2 = (float*)calloc(cfg->ff_dim,                       sizeof(float));

    size_t kv_cache_size = (size_t)cfg->n_layers * cfg->max_seq_len
                         * cfg->n_kv_heads * cfg->head_dim;
    s->k_cache = (float*)calloc(kv_cache_size, sizeof(float));
    s->v_cache = (float*)calloc(kv_cache_size, sizeof(float));

    if (!s->x||!s->xb||!s->q||!s->k||!s->v||!s->att||!s->logits
        ||!s->ff_buf||!s->ff_buf2||!s->k_cache||!s->v_cache) {
        fprintf(stderr,"OOM allocating run state\n"); return -1;
    }
    return 0;
}

static void free_run_state(RunState *s) {
    free(s->x); free(s->xb); free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits); free(s->ff_buf); free(s->ff_buf2);
    free(s->k_cache); free(s->v_cache);
}

/* ─────────────────────────── forward pass ──────────────────────────── */

static float* forward(Weights *w, Config *cfg, RunState *s, int token, int pos) {
    int dim    = cfg->dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;
    int q_dim  = cfg->n_heads    * cfg->head_dim;
    float scale = 1.0f / sqrtf((float)cfg->head_dim);

    /* 1. Token embedding lookup */
    float *emb = w->embed_tokens + (size_t)token * dim;
    memcpy(s->x, emb, dim * sizeof(float));

    /* 2. Transformer layers */
    for (int l=0; l<cfg->n_layers; l++) {

        /* Attention pre-norm */
        rmsnorm(s->xb, s->x, w->attn_norm[l], dim);

        /* QKV projections */
        memset(s->q, 0, q_dim  * sizeof(float));
        memset(s->k, 0, kv_dim * sizeof(float));
        memset(s->v, 0, kv_dim * sizeof(float));
        matmul_acc(s->q, s->xb, w->attn_q[l], q_dim,  dim);
        matmul_acc(s->k, s->xb, w->attn_k[l], kv_dim, dim);
        matmul_acc(s->v, s->xb, w->attn_v[l], kv_dim, dim);

        /* RoPE */
        rope(s->q, s->k, cfg->head_dim, cfg->n_heads, cfg->n_kv_heads,
             pos, cfg->rope_theta);

        /* Store KV in cache */
        size_t kv_offset = ((size_t)l * cfg->max_seq_len + pos) * kv_dim;
        memcpy(s->k_cache + kv_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->v_cache + kv_offset, s->v, kv_dim * sizeof(float));

        /* Grouped-Query Attention */
        memset(s->xb, 0, dim * sizeof(float));
        for (int h=0; h<cfg->n_heads; h++) {
            int    kv_head = h / cfg->gqa_factor;
            float *qh      = s->q  + h * cfg->head_dim;
            float *atth    = s->att + h * cfg->max_seq_len;

            for (int t=0; t<=pos; t++) {
                size_t kv_t = ((size_t)l * cfg->max_seq_len + t) * kv_dim
                            + (size_t)kv_head * cfg->head_dim;
                float *kt  = s->k_cache + kv_t;
                float dot  = 0.0f;
                for (int i=0; i<cfg->head_dim; i++) dot += qh[i]*kt[i];
                atth[t] = dot * scale;
            }
            softmax(atth, pos+1);

            float *out_h = s->xb + h * cfg->head_dim;
            for (int t=0; t<=pos; t++) {
                size_t kv_t = ((size_t)l * cfg->max_seq_len + t) * kv_dim
                            + (size_t)kv_head * cfg->head_dim;
                float *vt = s->v_cache + kv_t;
                float a   = atth[t];
                for (int i=0; i<cfg->head_dim; i++) out_h[i] += a * vt[i];
            }
        }

        /* Output projection + residual */
        float tmp[4096];
        matmul(tmp, s->xb, w->attn_o[l], dim, q_dim);
        for (int i=0; i<dim; i++) s->x[i] += tmp[i];

        /* FFN pre-norm */
        rmsnorm(s->xb, s->x, w->ff_norm[l], dim);

        /* SwiGLU FFN */
        matmul(s->ff_buf,  s->xb, w->ff_gate[l], cfg->ff_dim, dim);
        matmul(s->ff_buf2, s->xb, w->ff_up[l],   cfg->ff_dim, dim);
        silu(s->ff_buf, cfg->ff_dim);
        for (int i=0; i<cfg->ff_dim; i++) s->ff_buf[i] *= s->ff_buf2[i];
        matmul(tmp, s->ff_buf, w->ff_down[l], dim, cfg->ff_dim);
        for (int i=0; i<dim; i++) s->x[i] += tmp[i];
    }

    /* 3. Final norm */
    rmsnorm(s->x, s->x, w->norm, dim);

    /* 4. LM head */
    matmul(s->logits, s->x, w->lm_head, cfg->vocab_size, dim);

    return s->logits;
}

/* ─────────────────────────── sampling ─────────────────────────────── */

static int sample_argmax(float *logits, int n) {
    int best = 0;
    for (int i=1; i<n; i++) if (logits[i]>logits[best]) best=i;
    return best;
}

static int sample_temperature(float *logits, int n, float temp) {
    for (int i=0; i<n; i++) logits[i] /= temp;
    softmax(logits, n);
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i=0; i<n; i++) {
        cumsum += logits[i];
        if (r < cumsum) return i;
    }
    return n-1;
}

/* ─────────────────────────── timing helper ─────────────────────────── */

/* Returns wall-clock seconds as a double */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─────────────────────────── main ─────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <model_dir> \"<prompt>\" [max_tokens] [temperature]\n"
            "\n"
            "  model_dir      directory with model.safetensors and tokenizer.json\n"
            "  max_tokens     default 200\n"
            "  temperature    0 = greedy, >0 = sampling (default 0)\n"
            "\n"
            "BLAS backend compiled in: %s\n"
            "\n"
            "Build variants:\n"
            "  OpenBLAS:   gcc -O3 -march=native -o llama      llama.c -lm -lopenblas\n"
            "  Intel MKL:  gcc -O3 -march=native -DUSE_MKL -o llama_mkl llama.c -lm -lmkl_rt\n",
            argv[0], BLAS_BACKEND_NAME);
        return 1;
    }

    const char *model_dir = argv[1];
    const char *prompt    = argv[2];
    int   max_tokens  = argc>3 ? atoi(argv[3]) : 200;
    float temperature = argc>4 ? atof(argv[4]) : 0.0f;

    srand((unsigned)time(NULL));

    /* Banner */
    printf("╔══════════════════════════════════════╗\n");
    printf("║  llama.c  │  BLAS: %-17s║\n", BLAS_BACKEND_NAME);
    printf("╚══════════════════════════════════════╝\n");

    /* Config */
    Config cfg = llama32_1b_config();
    printf("[config] dim=%d layers=%d heads=%d kv_heads=%d ff=%d vocab=%d\n",
           cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
           cfg.ff_dim, cfg.vocab_size);

    /* Open safetensors */
    char st_path[512];
    const char *candidates[] = {
        "%s/model.safetensors",
        "%s/model-00001-of-00001.safetensors",
        "%s/model-00001-of-00002.safetensors",
        "%s",
        NULL
    };

    SafeTensors st;
    memset(&st, 0, sizeof(st));
    int opened = 0;
    for (int ci = 0; candidates[ci]; ci++) {
        snprintf(st_path, sizeof(st_path), candidates[ci], model_dir);
        struct stat _sb;
        if (stat(st_path, &_sb) != 0) continue;
        if (st_open(&st, st_path) == 0) { opened = 1; break; }
    }
    if (!opened) {
        fprintf(stderr,
            "Could not open model weights.\n"
            "Tried directory: %s\n"
            "Expected one of: model.safetensors, model-00001-of-00001.safetensors\n"
            "Or pass the .safetensors file path directly as the first argument.\n",
            model_dir);
        return 1;
    }

    /* Load weights */
    Weights w; memset(&w, 0, sizeof(w));
    if (load_weights(&w, &cfg, &st) < 0) return 1;
    st_close(&st);

    /* Tokenizer */
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    Tokenizer tok; memset(&tok, 0, sizeof(tok));
    if (tok_load(&tok, tok_path) < 0) return 1;

    /* Run state */
    RunState s; memset(&s, 0, sizeof(s));
    if (alloc_run_state(&s, &cfg) < 0) return 1;

    /* Encode prompt — prepend BOS */
    int ids[4096];
    ids[0] = tok.bos;
    int n_prompt = tok_encode(&tok, prompt, ids+1, 4095) + 1;
    printf("[encode] Prompt tokens: %d\n", n_prompt);
    printf("\n--- Output ---\n");
    fwrite(prompt, 1, strlen(prompt), stdout);
    fflush(stdout);

    /* Inference loop — timed */
    double t_prefill_start = now_sec();
    double t_first_token   = 0.0;
    int    n_generated     = 0;

    int next_token = 0;
    int total = n_prompt + max_tokens;
    if (total > cfg.max_seq_len) total = cfg.max_seq_len;

    for (int pos=0; pos<total; pos++) {
        int token = (pos < n_prompt) ? ids[pos] : next_token;
        float *logits = forward(&w, &cfg, &s, token, pos);

        if (pos == n_prompt - 1) {
            /* Last prefill step done — record time to first generated token */
            t_first_token = now_sec();
        }

        if (pos < n_prompt-1) continue; /* prefill, no output yet */

        /* Sample */
        if (temperature <= 0.0f)
            next_token = sample_argmax(logits, cfg.vocab_size);
        else
            next_token = sample_temperature(logits, cfg.vocab_size, temperature);

        if (next_token == tok.eos || next_token == tok.bos) {
            printf("\n[EOS]\n");
            break;
        }

        if (tok.vocab[next_token] && tok.vocab_lens[next_token] > 0) {
            fwrite(tok.vocab[next_token], 1, tok.vocab_lens[next_token], stdout);
            fflush(stdout);
        }
        n_generated++;
    }

    /* ── Timing report ──────────────────────────────────────────────── */
    double t_end = now_sec();
    double t_prefill  = t_first_token - t_prefill_start;   /* prompt processing */
    double t_generate = t_end - t_first_token;              /* token generation  */
    double tok_per_sec = (t_generate > 0 && n_generated > 0)
                         ? (double)n_generated / t_generate
                         : 0.0;

    printf("\n\n══════════════════════════════════════\n");
    printf("  Backend       : %s\n",    BLAS_BACKEND_NAME);
    printf("  Prompt tokens : %d\n",   n_prompt);
    printf("  Generated     : %d tokens\n", n_generated);
    printf("  Prefill time  : %.2f s  (%.1f tok/s)\n",
           t_prefill,
           (t_prefill > 0 && n_prompt > 0) ? (double)n_prompt / t_prefill : 0.0);
    printf("  Generate time : %.2f s\n", t_generate);
    printf("  Throughput    : %.2f tok/s\n", tok_per_sec);
    printf("══════════════════════════════════════\n");

    /* Cleanup */
    free_run_state(&s);
    tok_free(&tok);

    free(w.embed_tokens); free(w.norm); free(w.lm_head);
    for (int l=0; l<cfg.n_layers; l++) {
        free(w.attn_q[l]); free(w.attn_k[l]); free(w.attn_v[l]); free(w.attn_o[l]);
        free(w.ff_gate[l]); free(w.ff_up[l]); free(w.ff_down[l]);
        free(w.attn_norm[l]); free(w.ff_norm[l]);
    }
    free(w.attn_q); free(w.attn_k); free(w.attn_v); free(w.attn_o);
    free(w.ff_gate); free(w.ff_up); free(w.ff_down);
    free(w.attn_norm); free(w.ff_norm);

    return 0;
}
/*
 gcc -O3 -march=native -DUSE_MKL llama.c \
-I/opt/intel/oneapi/mkl/2026.0/include \
-L/opt/intel/oneapi/mkl/2026.0/lib \
-lmkl_rt -lm \
-o llama_mkl
 gcc -O3 -march=native -o llama_openblas nllama.c -lm -lopenblas       
*/