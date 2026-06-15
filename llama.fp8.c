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

#ifdef USE_MKL
  #include <mkl_cblas.h>
  #define BLAS_BACKEND_NAME "Intel MKL"
#else
  #include <cblas.h>
  #define BLAS_BACKEND_NAME "OpenBLAS"
#endif

static float g_fp8_lut[256];

static void init_fp8_lut(void) {
    for (int i = 0; i < 256; i++) {
        uint8_t v   = (uint8_t)i;
        int     sgn = v >> 7;
        int     exp = (v >> 3) & 0xF;
        int     man = v & 0x7;

        if ((v & 0x7F) == 0x7F) { g_fp8_lut[i] = 0.0f; continue; } /* NaN → 0 */

        float val;
        if (exp == 0) {
            /* subnormal: 2^(1-7) × (man / 8)  =  man × 2^(-9) */
            val = (float)man * (1.0f / 512.0f);
        } else {
            /* normal: (1 + man/8) × 2^(exp-7) */
            val = (1.0f + (float)man * (1.0f / 8.0f))
                * powf(2.0f, (float)(exp - 7));
        }
        g_fp8_lut[i] = sgn ? -val : val;
    }
}

/* ══════════════════════════ model config ══════════════════════════════ */

typedef struct {
    int   dim;
    int   n_layers;
    int   n_heads;
    int   n_kv_heads;
    int   ff_dim;
    int   vocab_size;
    int   max_seq_len;
    float rope_theta;
    int   head_dim;
    int   gqa_factor;
} Config;

static Config llama32_1b_config(void) {
    Config c = {0};
    c.dim         = 2048;
    c.n_layers    = 16;
    c.n_heads     = 32;
    c.n_kv_heads  = 8;
    c.ff_dim      = 8192;
    c.vocab_size  = 128256;
    c.max_seq_len = 2048;
    c.rope_theta  = 500000.0f;
    c.head_dim    = c.dim / c.n_heads;
    c.gqa_factor  = c.n_heads / c.n_kv_heads;
    return c;
}

static Config llama31_8b_config(void) {
    Config c = {0};
    c.dim         = 4096;
    c.n_layers    = 32;
    c.n_heads     = 32;
    c.n_kv_heads  = 8;
    c.ff_dim      = 14336;
    c.vocab_size  = 128256;
    c.max_seq_len = 2048;   /* full context is 131072; reduce for RAM */
    c.rope_theta  = 500000.0f;
    c.head_dim    = c.dim / c.n_heads;
    c.gqa_factor  = c.n_heads / c.n_kv_heads;
    return c;
}

/* ══════════════════════════ .llmbin loader ════════════════════════════
 *
 * The file is mmap'd.  All tensor data is already F32, 64-byte aligned,
 * so weight pointers are just offsets into the mapped region — zero copy.
 */

#define LLMBIN_MAGIC   0x4C4C4D42u
#define LLMBIN_VERSION 1u

#define LLMBIN_DTYPE_F32  0
#define LLMBIN_DTYPE_BF16 1
#define LLMBIN_DTYPE_F16  2
#define LLMBIN_DTYPE_F8_E4M3FN 3

#define LLMBIN_INDEX_ENTRY 344   /* bytes per tensor index entry */
#define LLMBIN_ALIGN        64

typedef struct {
    char     name[256];
    uint32_t dtype;
    uint32_t ndim;
    uint64_t shape[8];
    uint64_t offset;   /* from start of DATA section */
    uint64_t nbytes;
} LLMBinEntry;

typedef struct {
    void        *mmap_ptr;
    size_t       mmap_len;
    int          fd;
    uint32_t     n_tensors;
    LLMBinEntry *index;    /* points into mmap */
    uint8_t     *data;     /* start of DATA section */
} LLMBin;

static int llmbin_open(LLMBin *lb, const char *path) {
    lb->fd = open(path, O_RDONLY);
    if (lb->fd < 0) { perror(path); return -1; }

    struct stat sb;
    fstat(lb->fd, &sb);
    lb->mmap_len = (size_t)sb.st_size;
    lb->mmap_ptr = mmap(NULL, lb->mmap_len, PROT_READ, MAP_PRIVATE, lb->fd, 0);
    if (lb->mmap_ptr == MAP_FAILED) { perror("mmap"); return -1; }

    uint8_t *raw = (uint8_t*)lb->mmap_ptr;
    uint32_t magic   = *(uint32_t*)(raw + 0);
    uint32_t version = *(uint32_t*)(raw + 4);
    uint32_t n       = *(uint32_t*)(raw + 8);

    if (magic != LLMBIN_MAGIC) {
        fprintf(stderr, "llmbin: bad magic 0x%08X (expected 0x%08X)\n", magic, LLMBIN_MAGIC);
        return -1;
    }
    if (version != LLMBIN_VERSION) {
        fprintf(stderr, "llmbin: unsupported version %u\n", version);
        return -1;
    }

    lb->n_tensors = n;
    lb->index     = (LLMBinEntry*)(raw + 16);

    /* DATA section starts after index, aligned to LLMBIN_ALIGN */
    size_t index_end = 16 + (size_t)n * LLMBIN_INDEX_ENTRY;
    size_t data_off  = (index_end + LLMBIN_ALIGN - 1) & ~(size_t)(LLMBIN_ALIGN - 1);
    lb->data = raw + data_off;

    printf("[llmbin] Loaded %u tensors from %s\n", n, path);
    return 0;
}

static void llmbin_close(LLMBin *lb) {
    if (lb->mmap_ptr && lb->mmap_ptr != MAP_FAILED)
        munmap(lb->mmap_ptr, lb->mmap_len);
    if (lb->fd >= 0) close(lb->fd);
}

/* Returns direct pointer into mmap (zero-copy for F32 data) */
static float* llmbin_get(LLMBin *lb, const char *name) {
    for (uint32_t i = 0; i < lb->n_tensors; i++) {
        if (strcmp(lb->index[i].name, name) == 0) {
            if (lb->index[i].dtype != LLMBIN_DTYPE_F32) {
                fprintf(stderr, "[llmbin] tensor '%s' is not F32 — re-run convert_weights.py without --keep-dtype\n", name);
                return NULL;
            }
            return (float*)(lb->data + lb->index[i].offset);
        }
    }
    fprintf(stderr, "[llmbin] missing tensor: %s\n", name);
    return NULL;
}

//fp8
static uint8_t* llmbin_get_fp8(LLMBin *lb, const char *name) {
    for (uint32_t i = 0; i < lb->n_tensors; i++) {
        if (strcmp(lb->index[i].name, name) == 0) {
            uint32_t dt = lb->index[i].dtype;
            if (dt != LLMBIN_DTYPE_F8_E4M3FN) {
                fprintf(stderr,
                    "[llmbin] tensor '%s' dtype=%u is not FP8_E4M3FN\n",
                    name, dt);
                return NULL;
            }
            return lb->data + lb->index[i].offset;
        }
    }
    fprintf(stderr, "[llmbin] missing tensor: %s\n", name);
    return NULL;
}

static float* llmbin_get_fp8_as_f32(LLMBin *lb, const char *name, int n) {
    uint8_t *src = llmbin_get_fp8(lb, name);
    if (!src) return NULL;
    float *dst = (float*)malloc((size_t)n * sizeof(float));
    if (!dst) { fprintf(stderr, "OOM dequant %s\n", name); return NULL; }
    for (int i = 0; i < n; i++) dst[i] = g_fp8_lut[src[i]];
    return dst;
}

/* Auto-detect model size from embed_tokens shape */
static Config llmbin_detect_config(LLMBin *lb) {
    for (uint32_t i = 0; i < lb->n_tensors; i++) {
        if (strcmp(lb->index[i].name, "model.embed_tokens.weight") == 0) {
            uint64_t dim = lb->index[i].shape[1];
            printf("[llmbin] Detected embedding dim = %llu\n", (unsigned long long)dim);
            if (dim == 4096) {
                printf("[llmbin] → Using Llama 3.1-8B config\n");
                return llama31_8b_config();
            } else {
                printf("[llmbin] → Using Llama 3.2-1B config\n");
                return llama32_1b_config();
            }
        }
    }
    fprintf(stderr, "[llmbin] WARNING: could not detect config, defaulting to 1B\n");
    return llama32_1b_config();
}

/* ══════════════════════════ model weights ═════════════════════════════ */

typedef struct {
    float  *embed_tokens;
    float **attn_q, **attn_k, **attn_v, **attn_o;
    float **ff_gate, **ff_up, **ff_down;
    float **attn_norm, **ff_norm;
    float  *norm, *lm_head;
    int     owns_memory;  /* 1 = malloc'd (safetensors path), 0 = mmap pointers (llmbin) */
} Weights;

typedef struct {
    uint8_t  *embed_tokens;               /* vocab_size × dim,  FP8  */
    uint8_t **attn_q, **attn_k, **attn_v, **attn_o;
    uint8_t **ff_gate, **ff_up, **ff_down;
    float   **attn_norm, **ff_norm;       /* dim-length vectors, F32 (dequanted at load) */
    float    *norm;                       /* final norm,         F32 (dequanted at load) */
    uint8_t  *lm_head;                    /* vocab_size × dim,  FP8  */
    int       lm_head_tied;               /* 1 = same ptr as embed_tokens */
    int       owns_norm_memory;           /* norm/attn_norm/ff_norm were malloc'd */
} Weight8;

/* ── llmbin path: zero-copy pointers ── */
static int load_weights_llmbin(Weights *w, Config *cfg, LLMBin *lb) {
    char name[256];

#define GET(dst, nm) do { \
    (dst) = llmbin_get(lb, (nm)); \
    if (!(dst)) { fprintf(stderr, "Missing tensor: %s\n", (nm)); return -1; } \
} while(0)

    GET(w->embed_tokens, "model.embed_tokens.weight");

    w->attn_q    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_k    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_v    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_o    = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_gate   = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_up     = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_down   = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->attn_norm = (float**)calloc(cfg->n_layers, sizeof(float*));
    w->ff_norm   = (float**)calloc(cfg->n_layers, sizeof(float*));

    for (int l = 0; l < cfg->n_layers; l++) {
#define LG(dst, fmt) snprintf(name,sizeof(name),fmt,l); GET(dst[l],name)
        LG(w->attn_q,    "model.layers.%d.self_attn.q_proj.weight");
        LG(w->attn_k,    "model.layers.%d.self_attn.k_proj.weight");
        LG(w->attn_v,    "model.layers.%d.self_attn.v_proj.weight");
        LG(w->attn_o,    "model.layers.%d.self_attn.o_proj.weight");
        LG(w->ff_gate,   "model.layers.%d.mlp.gate_proj.weight");
        LG(w->ff_up,     "model.layers.%d.mlp.up_proj.weight");
        LG(w->ff_down,   "model.layers.%d.mlp.down_proj.weight");
        LG(w->attn_norm, "model.layers.%d.input_layernorm.weight");
        LG(w->ff_norm,   "model.layers.%d.post_attention_layernorm.weight");
#undef LG
    }

    GET(w->norm, "model.norm.weight");

    float *lm = llmbin_get(lb, "lm_head.weight");
    if (lm) {
        w->lm_head = lm;
        printf("[weights] lm_head: separate tensor\n");
    } else {
        /* tied weights — point to embed_tokens */
        w->lm_head = w->embed_tokens;
        printf("[weights] lm_head: tied to embed_tokens\n");
    }

    w->owns_memory = 0;
    printf("[weights] All weights loaded (zero-copy from mmap).\n");
    return 0;
#undef GET
}

/* ═══════════════════════ load_weights_llmbin (FP8) ══════════════════ */
static int load_weight8_llmbin(Weight8 *w, Config *cfg, LLMBin *lb) {
    char name[256];

#define GET8(dst, nm) do { \
    (dst) = llmbin_get_fp8(lb, (nm)); \
    if (!(dst)) { fprintf(stderr, "Missing FP8 tensor: %s\n", (nm)); return -1; } \
} while(0)

#define GETF(dst, nm, n) do { \
    (dst) = llmbin_get_fp8_as_f32(lb, (nm), (n)); \
    if (!(dst)) { fprintf(stderr, "Missing norm tensor: %s\n", (nm)); return -1; } \
} while(0)

    GET8(w->embed_tokens, "model.embed_tokens.weight");

    w->attn_q    = (uint8_t**)calloc(cfg->n_layers, sizeof(uint8_t*));
    w->attn_k    = (uint8_t**)calloc(cfg->n_layers, sizeof(uint8_t*));
    w->attn_v    = (uint8_t**)calloc(cfg->n_layers, sizeof(uint8_t*));
    w->attn_o    = (uint8_t**)calloc(cfg->n_layers, sizeof(uint8_t*));
    w->ff_gate   = (uint8_t**)calloc(cfg->n_layers, sizeof(uint8_t*));
    w->ff_up     = (uint8_t**)calloc(cfg->n_layers, sizeof(uint8_t*));
    w->ff_down   = (uint8_t**)calloc(cfg->n_layers, sizeof(uint8_t*));
    w->attn_norm = (float**)  calloc(cfg->n_layers, sizeof(float*));
    w->ff_norm   = (float**)  calloc(cfg->n_layers, sizeof(float*));

    int dim = cfg->dim;

    for (int l = 0; l < cfg->n_layers; l++) {
        int q_dim  = cfg->n_heads    * cfg->head_dim;
        int kv_dim = cfg->n_kv_heads * cfg->head_dim;

#define LG8(dst, fmt, rows) \
    snprintf(name, sizeof(name), fmt, l); GET8(dst[l], name); (void)(rows)
#define LGF(dst, fmt, n) \
    snprintf(name, sizeof(name), fmt, l); GETF(dst[l], name, n)

        LG8(w->attn_q,    "model.layers.%d.self_attn.q_proj.weight", q_dim  * dim);
        LG8(w->attn_k,    "model.layers.%d.self_attn.k_proj.weight", kv_dim * dim);
        LG8(w->attn_v,    "model.layers.%d.self_attn.v_proj.weight", kv_dim * dim);
        LG8(w->attn_o,    "model.layers.%d.self_attn.o_proj.weight", dim    * q_dim);
        LG8(w->ff_gate,   "model.layers.%d.mlp.gate_proj.weight",    cfg->ff_dim * dim);
        LG8(w->ff_up,     "model.layers.%d.mlp.up_proj.weight",      cfg->ff_dim * dim);
        LG8(w->ff_down,   "model.layers.%d.mlp.down_proj.weight",    dim * cfg->ff_dim);
        LGF(w->attn_norm, "model.layers.%d.input_layernorm.weight",           dim);
        LGF(w->ff_norm,   "model.layers.%d.post_attention_layernorm.weight",  dim);
#undef LG8
#undef LGF
    }

    GETF(w->norm, "model.norm.weight", dim);

    /* lm_head — may be tied to embed_tokens */
    uint8_t *lm = llmbin_get_fp8(lb, "lm_head.weight");
    if (lm) {
        w->lm_head     = lm;
        w->lm_head_tied = 0;
        printf("[weights] lm_head: separate FP8 tensor\n");
    } else {
        w->lm_head      = w->embed_tokens;
        w->lm_head_tied = 1;
        printf("[weights] lm_head: tied to embed_tokens\n");
    }

    w->owns_norm_memory = 1;   /* attn_norm/ff_norm/norm were malloc'd by GETF */
    printf("[weights] All FP8 weights loaded (zero-copy from mmap).\n");
    return 0;

#undef GET8
#undef GETF
}

static void free_weights(Weights *w, Config *cfg) {
    if (!w->owns_memory) {
        /* llmbin: only free the pointer arrays, not the data */
        free(w->attn_q); free(w->attn_k); free(w->attn_v); free(w->attn_o);
        free(w->ff_gate); free(w->ff_up); free(w->ff_down);
        free(w->attn_norm); free(w->ff_norm);
        return;
    }
    free(w->embed_tokens); free(w->norm);
    /* lm_head might be tied, free only if different pointer */
    if (w->lm_head != w->embed_tokens) free(w->lm_head);
    for(int l=0;l<cfg->n_layers;l++){
        free(w->attn_q[l]);free(w->attn_k[l]);free(w->attn_v[l]);free(w->attn_o[l]);
        free(w->ff_gate[l]);free(w->ff_up[l]);free(w->ff_down[l]);
        free(w->attn_norm[l]);free(w->ff_norm[l]);
    }
    free(w->attn_q);free(w->attn_k);free(w->attn_v);free(w->attn_o);
    free(w->ff_gate);free(w->ff_up);free(w->ff_down);
    free(w->attn_norm);free(w->ff_norm);
}

static void free_weight8(Weight8 *w, Config *cfg) {
    /* FP8 data ptrs are into the mmap — do NOT free them */
    free(w->attn_q); free(w->attn_k); free(w->attn_v); free(w->attn_o);
    free(w->ff_gate); free(w->ff_up); free(w->ff_down);

    if (w->owns_norm_memory) {
        for (int l = 0; l < cfg->n_layers; l++) {
            free(w->attn_norm[l]);
            free(w->ff_norm[l]);
        }
        free(w->norm);
    }
    free(w->attn_norm); free(w->ff_norm);
}
/* ══════════════════════════ tokenizer ═════════════════════════════════
 *
 * Bugs fixed vs original:
 *
 * 1. find_merges_array() was searching for "merges" inside a manually
 *    walked "model" object, but would silently fail if any key before
 *    "merges" had a complex value (arrays, nested objects) — the skip
 *    logic didn't handle nested structures.  Now we just strstr for
 *    the well-known key path and skip properly with skip_json_value().
 *
 * 2. The merge lookup built the merged token string from the encoded
 *    (GPT-2 byte-level) form, but the result lookup used hm_get on
 *    the SAME encoded string.  This is correct, but only if enc_vocab
 *    entries for multi-byte tokens are stored in their encoded form.
 *    The original code stored enc_vocab[id] = strdup(enc_tok) which
 *    IS the encoded form, so that part was fine — the real culprit was
 *    find_merges_array silently returning NULL on most real files.
 *
 * 3. Added fallback: if merges section is not inside "model", also
 *    search at the top level (some tokenizer.json variants differ).
 */

static void build_byte_decoder(uint32_t enc[256], uint8_t dec[1024]) {
    memset(dec, 0xff, 1024);
    int n=0;
    for(int b=0;b<256;b++){
        int p=(b>=33&&b<=126)||(b>=161&&b<=172)||(b>=174&&b<=255);
        enc[b] = p ? (uint32_t)b : (uint32_t)(0x100+n);
        if(!p && n<1024){ dec[n]=(uint8_t)b; n++; }
    }
}

static int decode_gpt2_token(const char *tok_str, uint8_t *out, int max_out, uint8_t byte_dec[1024]) {
    int out_len=0;
    const uint8_t *p=(const uint8_t*)tok_str;
    while(*p && out_len<max_out){
        uint32_t cp;
        if     (*p<0x80){cp=*p++;}
        else if((*p&0xe0)==0xc0){cp=(*p&0x1f)<<6;p++;cp|=(*p&0x3f);p++;}
        else if((*p&0xf0)==0xe0){cp=(*p&0x0f)<<12;p++;cp|=(*p&0x3f)<<6;p++;cp|=(*p&0x3f);p++;}
        else{p++;continue;}
        uint8_t raw;
        if(cp<0x100) raw=(uint8_t)cp;
        else if((cp-0x100)<1024&&byte_dec[cp-0x100]!=0xff) raw=byte_dec[cp-0x100];
        else raw='?';
        out[out_len++]=raw;
    }
    return out_len;
}

typedef struct { int a,b,result; } Merge;
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

#define HASHMAP_CAP (1<<18)
typedef struct { const char *key; int val; } HMEntry;
static HMEntry *g_hm=NULL;

static uint32_t hm_hash(const char *s){
    uint32_t h=2166136261u; while(*s){h^=(uint8_t)*s++;h*=16777619u;} return h;
}
static void hm_set(const char *key,int val){
    uint32_t h=hm_hash(key)&(HASHMAP_CAP-1);
    while(g_hm[h].key&&strcmp(g_hm[h].key,key)!=0) h=(h+1)&(HASHMAP_CAP-1);
    g_hm[h].key=key; g_hm[h].val=val;
}
static int hm_get(const char *key,int def){
    uint32_t h=hm_hash(key)&(HASHMAP_CAP-1);
    while(g_hm[h].key){if(!strcmp(g_hm[h].key,key))return g_hm[h].val; h=(h+1)&(HASHMAP_CAP-1);}
    return def;
}

static int read_json_str(const char **pp, char *out, int max){
    const char *p=*pp; if(*p!='"') return -1; p++;
    int j=0;
    while(*p&&*p!='"'){
        if(*p=='\\'){
            p++;
            switch(*p){
                case 'n':if(j<max-1)out[j++]='\n';break;
                case 't':if(j<max-1)out[j++]='\t';break;
                case 'r':if(j<max-1)out[j++]='\r';break;
                case '\\':if(j<max-1)out[j++]='\\';break;
                case '"':if(j<max-1)out[j++]='"';break;
                case '/':if(j<max-1)out[j++]='/';break;
                case 'u':{
                    char hex[5]={0}; for(int h=0;h<4&&p[1];h++) hex[h]=*++p;
                    uint32_t cp=(uint32_t)strtoul(hex,NULL,16);
                    if(cp<0x80){if(j<max-1)out[j++]=(char)cp;}
                    else if(cp<0x800){if(j<max-2){out[j++]=0xc0|(cp>>6);out[j++]=0x80|(cp&0x3f);}}
                    else{if(j<max-3){out[j++]=0xe0|(cp>>12);out[j++]=0x80|((cp>>6)&0x3f);out[j++]=0x80|(cp&0x3f);}}
                    break;
                }
                default:if(j<max-1)out[j++]=*p;break;
            }
        } else { if(j<max-1)out[j++]=*p; }
        p++;
    }
    out[j]=0; if(*p=='"')p++; *pp=p; return j;
}

static const char* skip_json_value(const char *p){
    while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
    if(*p=='"'){
        p++; while(*p&&*p!='"'){if(*p=='\\')p++;p++;}
        if(*p=='"')p++; return p;
    }
    if(*p=='{'||*p=='['){
        char open=*p, close=(*p=='{') ? '}' : ']';
        int depth=1; p++;
        while(*p&&depth>0){
            if(*p=='\\'){p+=2;continue;}
            if(*p=='"'){p++;while(*p&&*p!='"'){if(*p=='\\')p++;p++;}if(*p=='"')p++;continue;}
            if(*p==open)depth++; if(*p==close)depth--;
            p++;
        }
        return p;
    }
    while(*p&&*p!=','&&*p!='}'&&*p!=']'&&*p!=' '&&*p!='\n'&&*p!='\r') p++;
    return p;
}

/*
 * find_merges_array — fixed version
 * ──────────────────────────────────
 * The original used a hand-rolled object walker that failed silently
 * whenever a value before "merges" was an array or nested object
 * (it only handled strings and numbers).
 *
 * New approach:
 *   1. Find the "model" key anywhere in the JSON.
 *   2. Walk its object keys using skip_json_value() for any value type.
 *   3. Also try searching directly at the top level (some variants).
 */
static const char* find_merges_array_in_obj(const char *obj_start) {
    /* obj_start points AFTER the opening '{' of the object */
    const char *p = obj_start;
    for (;;) {
        while (*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
        if (*p=='}' || !*p) break;
        if (*p==',') { p++; continue; }
        if (*p!='"') { p++; continue; }

        /* read key */
        char key[128]={0}; int ki=0; p++;
        while(*p&&*p!='"'){ if(*p=='\\')p++; if(ki<127)key[ki++]=*p; p++; }
        if(*p=='"')p++;
        key[ki]=0;

        while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
        if(*p!=':') break; p++;
        while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;

        if(strcmp(key,"merges")==0 && *p=='[') return p;

        p = skip_json_value(p);
    }
    return NULL;
}

static const char* find_merges_array(const char *js) {
    /* Strategy 1: look inside "model": { ... "merges": [...] } */
    const char *model_key = js;
    while ((model_key = strstr(model_key, "\"model\"")) != NULL) {
        const char *p = model_key + 7;
        while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
        if(*p!=':') { model_key++; continue; }
        p++;
        while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
        if(*p!='{') { model_key++; continue; }
        p++; /* skip '{' */
        const char *arr = find_merges_array_in_obj(p);
        if (arr) return arr;
        model_key++;
    }

    /* Strategy 2: top-level "merges" key */
    const char *top = js;
    if(*top=='{') top++;
    const char *arr = find_merges_array_in_obj(top);
    if (arr) return arr;

    return NULL;
}

static int tok_load(Tokenizer *tok, const char *path){
    FILE *f=fopen(path,"r"); if(!f){perror(path);return -1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char *js=(char*)malloc(sz+1); if(!js){fclose(f);return -1;}
    if(fread(js,1,sz,f)!=(size_t)sz) fprintf(stderr,"Warning: short read\n");
    js[sz]=0; fclose(f);

    build_byte_decoder(tok->byte_enc,tok->byte_dec);

    tok->vocab_size=128256;
    tok->vocab     =(char**)calloc(tok->vocab_size,sizeof(char*));
    tok->vocab_lens=(int*)  calloc(tok->vocab_size,sizeof(int));
    tok->enc_vocab =(char**)calloc(tok->vocab_size,sizeof(char*));
    tok->bos=128000; tok->eos=128009;

    /* ── 1. added_tokens (special tokens) ── */
    const char *at=strstr(js,"\"added_tokens\"");
    if(at){
        const char *p=strchr(at,'['); if(p){ p++;
            while(*p){
                while(*p==' '||*p=='\n'||*p=='\r'||*p=='\t') p++;
                if(*p==']') break; if(*p==','){p++;continue;}
                if(*p!='{'){p++;continue;}
                int id=-1; char content[256]={0};
                const char *oe=p; int depth=1; oe++;
                while(*oe&&depth>0){if(*oe=='{')depth++;else if(*oe=='}')depth--;oe++;}
                char obj[4096]; int olen=(int)(oe-p); if(olen>4095)olen=4095;
                memcpy(obj,p,olen); obj[olen]=0;
                char *idp=strstr(obj,"\"id\"");
                if(idp){idp+=4;while(*idp==' '||*idp==':')idp++;id=atoi(idp);}
                char *cp2=strstr(obj,"\"content\"");
                if(cp2){cp2+=9;while(*cp2==' '||*cp2==':')cp2++;const char *cp3=(const char*)cp2;read_json_str(&cp3,content,sizeof(content));}
                if(id>=0&&id<tok->vocab_size&&content[0]){
                    free(tok->vocab[id]);
                    tok->vocab[id]=strdup(content); tok->vocab_lens[id]=(int)strlen(content);
                    tok->enc_vocab[id]=strdup(content);
                }
                p=oe;
            }
        }
    }

    /* ── 2. model.vocab ── */
    const char *model_sec=strstr(js,"\"model\""); if(!model_sec){fprintf(stderr,"No 'model'\n");free(js);return -1;}
    const char *vocab_sec=strstr(model_sec,"\"vocab\""); if(!vocab_sec){fprintf(stderr,"No vocab\n");free(js);return -1;}
    const char *vp=strchr(vocab_sec,'{'); if(!vp){free(js);return -1;} vp++;
    int loaded=0;
    while(*vp){
        while(*vp==' '||*vp=='\n'||*vp=='\r'||*vp=='\t') vp++;
        if(*vp=='}') break; if(*vp==','){vp++;continue;} if(*vp!='"'){vp++;continue;}
        char enc_tok[512]; read_json_str(&vp,enc_tok,sizeof(enc_tok));
        while(*vp==' '||*vp=='\n'||*vp=='\r'||*vp=='\t') vp++;
        if(*vp!=':') continue; vp++;
        while(*vp==' ') vp++;
        int id=0,neg=0; if(*vp=='-'){neg=1;vp++;}
        if(*vp<'0'||*vp>'9') continue;
        while(*vp>='0'&&*vp<='9'){id=id*10+(*vp-'0');vp++;} if(neg)id=-id;
        if(id>=0&&id<tok->vocab_size&&!tok->enc_vocab[id]){
            uint8_t raw[512]; int rlen=decode_gpt2_token(enc_tok,raw,sizeof(raw)-1,tok->byte_dec); raw[rlen]=0;
            tok->enc_vocab[id]=strdup(enc_tok);
            tok->vocab[id]=(char*)malloc(rlen+1); if(tok->vocab[id]){memcpy(tok->vocab[id],raw,rlen+1);tok->vocab_lens[id]=rlen;}
            loaded++;
        }
    }
    printf("[tokenizer] vocab entries: %d\n",loaded);

    /* ── 3. merges (fixed) ── */
    int merge_cap=300000;
    tok->merges=(Merge*)malloc(merge_cap*sizeof(Merge)); tok->n_merges=0;
    g_hm=(HMEntry*)calloc(HASHMAP_CAP,sizeof(HMEntry));
    for(int i=0;i<tok->vocab_size;i++) if(tok->enc_vocab[i]) hm_set(tok->enc_vocab[i],i);

    const char *merges_arr=find_merges_array(js);
    if(!merges_arr){
        fprintf(stderr,"[tokenizer] ERROR: could not find merges array — encoding will be character-level only!\n");
    } else {
        const char *mp=merges_arr; if(*mp=='[') mp++;
        int skipped=0;
        while(*mp&&tok->n_merges<merge_cap){
            while(*mp==' '||*mp=='\n'||*mp=='\r'||*mp=='\t') mp++;
            if(*mp==']') break; if(*mp==','){mp++;continue;} if(*mp!='"'){mp++;continue;}
            char ms[512]; read_json_str(&mp,ms,sizeof(ms));
            char *space=strchr(ms,' '); if(!space){skipped++;continue;}
            *space=0;
            int ia=hm_get(ms,-1), ib=hm_get(space+1,-1);
            if(ia<0||ib<0){skipped++;continue;}
            char merged[1024]; snprintf(merged,sizeof(merged),"%s%s",ms,space+1);
            int ir=hm_get(merged,-1);
            if(ir<0){skipped++;continue;}
            tok->merges[tok->n_merges++]=(Merge){ia,ib,ir};
        }
        if(skipped>0) printf("[tokenizer] merges skipped (not in vocab): %d\n",skipped);
    }
    printf("[tokenizer] merges loaded: %d\n",tok->n_merges);

    /* ── 4. fill gaps ── */
    for(int i=0;i<tok->vocab_size;i++){
        if(!tok->vocab[i]){char tmp[32];snprintf(tmp,sizeof(tmp),"<unk%d>",i);tok->vocab[i]=strdup(tmp);tok->vocab_lens[i]=(int)strlen(tmp);}
    }
    printf("[tokenizer] bos=%d eos=%d\n",tok->bos,tok->eos);
    free(js); return 0;
}

static int byte_to_gpt2_str(uint8_t byte,uint32_t byte_enc[256],char *out){
    uint32_t enc=byte_enc[byte]; int len=0;
    if(enc<0x80){out[len++]=(char)enc;}
    else if(enc<0x800){out[len++]=(char)(0xc0|(enc>>6));out[len++]=(char)(0x80|(enc&0x3f));}
    else{out[len++]=(char)(0xe0|(enc>>12));out[len++]=(char)(0x80|((enc>>6)&0x3f));out[len++]=(char)(0x80|(enc&0x3f));}
    out[len]=0; return len;
}

// static int tok_encode(Tokenizer *tok, const char *text, int *ids, int max_ids){
//     int seq_cap=65536; int *seq=(int*)malloc(seq_cap*sizeof(int)); if(!seq) return 0;
//     int seq_len=0;
//     const uint8_t *p=(const uint8_t*)text;
//     while(*p&&seq_len<seq_cap-1){
//         uint32_t cp; int cp_bytes;
//         if     ((*p&0x80)==0x00){cp=*p;cp_bytes=1;}
//         else if((*p&0xe0)==0xc0){cp=((uint32_t)(p[0]&0x1f)<<6)|(p[1]&0x3f);cp_bytes=2;}
//         else if((*p&0xf0)==0xe0){cp=((uint32_t)(p[0]&0x0f)<<12)|((uint32_t)(p[1]&0x3f)<<6)|(p[2]&0x3f);cp_bytes=3;}
//         else if((*p&0xf8)==0xf0){cp=((uint32_t)(p[0]&0x07)<<18)|((uint32_t)(p[1]&0x3f)<<12)|((uint32_t)(p[2]&0x3f)<<6)|(p[3]&0x3f);cp_bytes=4;}
//         else{cp=*p;cp_bytes=1;}
//         uint8_t utf8[4]; int nb=0;
//         if(cp<0x80){utf8[0]=(uint8_t)cp;nb=1;}
//         else if(cp<0x800){utf8[0]=0xc0|(cp>>6);utf8[1]=0x80|(cp&0x3f);nb=2;}
//         else if(cp<0x10000){utf8[0]=0xe0|(cp>>12);utf8[1]=0x80|((cp>>6)&0x3f);utf8[2]=0x80|(cp&0x3f);nb=3;}
//         else{utf8[0]=0xf0|(cp>>18);utf8[1]=0x80|((cp>>12)&0x3f);utf8[2]=0x80|((cp>>6)&0x3f);utf8[3]=0x80|(cp&0x3f);nb=4;}
//         char gpt2_str[32]; int glen=0;
//         for(int b=0;b<nb;b++) glen+=byte_to_gpt2_str(utf8[b],tok->byte_enc,gpt2_str+glen);
//         gpt2_str[glen]=0;
//         int id=hm_get(gpt2_str,-1); if(id>=0) seq[seq_len++]=id;
//         p+=cp_bytes;
//     }
//     /* BPE merges */
//     for(int m=0;m<tok->n_merges&&seq_len>1;m++){
//         int a=tok->merges[m].a,b=tok->merges[m].b,c=tok->merges[m].result,write=0;
//         for(int r=0;r<seq_len;r++){
//             if(r<seq_len-1&&seq[r]==a&&seq[r+1]==b){seq[write++]=c;r++;}
//             else seq[write++]=seq[r];
//         }
//         seq_len=write;
//     }
//     int n=(seq_len<max_ids-1)?seq_len:max_ids-1;
//     memcpy(ids,seq,n*sizeof(int)); free(seq); return n;
// }

/* ── FIXED tok_encode: rank-based BPE for tiktoken (Llama 3) ──────────────
 *
 * Llama 3 uses tiktoken, not GPT-2-style ordered merges.  The rule is:
 *   - At each step, scan all adjacent pairs in the current sequence.
 *   - For each pair, concatenate their encoded strings and look up the result
 *     in the vocab hashmap.
 *   - Merge the pair whose RESULT has the lowest token ID (rank).
 *   - Repeat until no pair can be merged.
 *
 * This replaces the broken "for each merge in order, apply everywhere" loop.
 * Complexity: O(n²) merges × O(n) scan = O(n³) worst case, but n is small
 * (single prompt) and the inner loop exits early when no merge is found.
 */
static int tok_encode(Tokenizer *tok, const char *text, int *ids, int max_ids){
    int seq_cap = 65536;
    int *seq = (int*)malloc(seq_cap * sizeof(int));
    if (!seq) return 0;
    int seq_len = 0;

    /* ── 1. Initial encoding: text → individual byte tokens ── */
    const uint8_t *p = (const uint8_t*)text;
    while (*p && seq_len < seq_cap - 1) {
        uint32_t cp; int cp_bytes;
        if      ((*p & 0x80) == 0x00) { cp = *p;                                                                                cp_bytes = 1; }
        else if ((*p & 0xe0) == 0xc0) { cp = ((uint32_t)(p[0]&0x1f)<<6)|(p[1]&0x3f);                                          cp_bytes = 2; }
        else if ((*p & 0xf0) == 0xe0) { cp = ((uint32_t)(p[0]&0x0f)<<12)|((uint32_t)(p[1]&0x3f)<<6)|(p[2]&0x3f);             cp_bytes = 3; }
        else if ((*p & 0xf8) == 0xf0) { cp = ((uint32_t)(p[0]&0x07)<<18)|((uint32_t)(p[1]&0x3f)<<12)|((uint32_t)(p[2]&0x3f)<<6)|(p[3]&0x3f); cp_bytes = 4; }
        else { cp = *p; cp_bytes = 1; }

        uint8_t utf8[4]; int nb = 0;
        if      (cp < 0x80)    { utf8[0] = (uint8_t)cp; nb = 1; }
        else if (cp < 0x800)   { utf8[0]=0xc0|(cp>>6);   utf8[1]=0x80|(cp&0x3f); nb=2; }
        else if (cp < 0x10000) { utf8[0]=0xe0|(cp>>12);  utf8[1]=0x80|((cp>>6)&0x3f); utf8[2]=0x80|(cp&0x3f); nb=3; }
        else                   { utf8[0]=0xf0|(cp>>18);  utf8[1]=0x80|((cp>>12)&0x3f); utf8[2]=0x80|((cp>>6)&0x3f); utf8[3]=0x80|(cp&0x3f); nb=4; }

        char gpt2_str[32]; int glen = 0;
        for (int b = 0; b < nb; b++)
            glen += byte_to_gpt2_str(utf8[b], tok->byte_enc, gpt2_str + glen);
        gpt2_str[glen] = 0;

        int id = hm_get(gpt2_str, -1);
        if (id >= 0) seq[seq_len++] = id;
        p += cp_bytes;
    }

    /* ── 2. Rank-based BPE ────────────────────────────────────────────────
     *
     * At each round, find the adjacent pair (seq[i], seq[i+1]) whose merged
     * token has the SMALLEST id.  Apply that one merge, then re-scan.
     * Repeat until nothing can be merged.
     */
    char merged_key[2048];
    for (;;) {
        if (seq_len <= 1) break;

        int best_pos = -1;
        int best_id  = tok->vocab_size; /* sentinel: larger than any real id */

        for (int i = 0; i < seq_len - 1; i++) {
            const char *a = tok->enc_vocab[seq[i]];
            const char *b = tok->enc_vocab[seq[i+1]];
            if (!a || !b) continue;

            int alen = (int)strlen(a), blen = (int)strlen(b);
            if (alen + blen >= (int)sizeof(merged_key)) continue;

            memcpy(merged_key,       a, alen);
            memcpy(merged_key+alen,  b, blen);
            merged_key[alen+blen] = '\0';

            int id = hm_get(merged_key, -1);
            if (id >= 0 && id < best_id) {
                best_id  = id;
                best_pos = i;
            }
        }

        if (best_pos < 0) break; /* no pair can be merged → done */

        /* Apply the single best merge */
        seq[best_pos] = best_id;
        /* Shift everything after the consumed pair left by one */
        for (int i = best_pos + 1; i < seq_len - 1; i++)
            seq[i] = seq[i+1];
        seq_len--;
    }

    int n = (seq_len < max_ids - 1) ? seq_len : max_ids - 1;
    memcpy(ids, seq, n * sizeof(int));
    free(seq);
    return n;
}

static void tok_free(Tokenizer *tok){
    for(int i=0;i<tok->vocab_size;i++){free(tok->vocab[i]);free(tok->enc_vocab[i]);}
    free(tok->vocab);free(tok->vocab_lens);free(tok->enc_vocab);free(tok->merges);
    free(g_hm);g_hm=NULL;
}

/* ══════════════════════════ math helpers ══════════════════════════════ */

static void rmsnorm(float *out,const float *x,const float *w,int dim){
    float ss=0.0f; for(int i=0;i<dim;i++)ss+=x[i]*x[i];
    ss=1.0f/sqrtf(ss/(float)dim+1e-5f);
    for(int i=0;i<dim;i++)out[i]=w[i]*(ss*x[i]);
}

static void matmul(float *out,const float *x,const float *W,int out_dim,int in_dim){
    cblas_sgemv(CblasRowMajor,CblasNoTrans,out_dim,in_dim,1.0f,W,in_dim,x,1,0.0f,out,1);
}
static void matmul_acc(float *out,const float *x,const float *W,int out_dim,int in_dim){
    cblas_sgemv(CblasRowMajor,CblasNoTrans,out_dim,in_dim,1.0f,W,in_dim,x,1,1.0f,out,1);
}

static void fp8_sgemv(float * restrict out,
                      const uint8_t * restrict W,
                      const float   * restrict x,
                      int out_dim, int in_dim)
{
    for (int i = 0; i < out_dim; i++) {
        const uint8_t *row = W + (size_t)i * in_dim;
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++)
            sum += g_fp8_lut[row[j]] * x[j];
        out[i] = sum;
    }
}

static void fp8_sgemv_acc(float * restrict out,
                          const uint8_t * restrict W,
                          const float   * restrict x,
                          int out_dim, int in_dim)
{
    for (int i = 0; i < out_dim; i++) {
        const uint8_t *row = W + (size_t)i * in_dim;
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++)
            sum += g_fp8_lut[row[j]] * x[j];
        out[i] += sum;
    }
}

static void softmax(float *x,int n){
    float mx=x[0]; for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];
    float s=0.0f; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    for(int i=0;i<n;i++)x[i]/=s;
}
static void silu(float *x,int n){ for(int i=0;i<n;i++)x[i]=x[i]/(1.0f+expf(-x[i])); }

static void rope(float *q,float *k,int head_dim,int n_heads,int n_kv_heads,int pos,float theta){
    int half=head_dim/2;
    for(int h=0;h<n_heads;h++){
        float *qh=q+h*head_dim;
        for(int i=0;i<half;i++){
            float freq=1.0f/powf(theta,(float)(2*i)/(float)head_dim);
            float angle=(float)pos*freq, c=cosf(angle), s=sinf(angle);
            float q0=qh[i],q1=qh[i+half]; qh[i]=q0*c-q1*s; qh[i+half]=q0*s+q1*c;
        }
    }
    for(int h=0;h<n_kv_heads;h++){
        float *kh=k+h*head_dim;
        for(int i=0;i<half;i++){
            float freq=1.0f/powf(theta,(float)(2*i)/(float)head_dim);
            float angle=(float)pos*freq, c=cosf(angle), s=sinf(angle);
            float k0=kh[i],k1=kh[i+half]; kh[i]=k0*c-k1*s; kh[i+half]=k0*s+k1*c;
        }
    }
}

/* ══════════════════════════ run state ═════════════════════════════════ */

typedef struct {
    float *x,*xb,*q,*k,*v,*att,*logits,*ff_buf,*ff_buf2;
    float *k_cache,*v_cache;
    float *tmp;  /* scratch for output projection: max(dim, ff_dim) */
    int    n_layers,n_kv_heads,head_dim,max_seq;
} RunState;

static int alloc_run_state(RunState *s, Config *cfg){
    int dim=cfg->dim, kv_dim=cfg->n_kv_heads*cfg->head_dim, q_dim=cfg->n_heads*cfg->head_dim;
    s->n_layers=cfg->n_layers; s->n_kv_heads=cfg->n_kv_heads;
    s->head_dim=cfg->head_dim; s->max_seq=cfg->max_seq_len;
    s->x      =(float*)calloc(dim,sizeof(float));
    s->xb     =(float*)calloc(dim,sizeof(float));
    s->q      =(float*)calloc(q_dim,sizeof(float));
    s->k      =(float*)calloc(kv_dim,sizeof(float));
    s->v      =(float*)calloc(kv_dim,sizeof(float));
    s->att    =(float*)calloc(cfg->n_heads*(size_t)cfg->max_seq_len,sizeof(float));
    s->logits =(float*)calloc(cfg->vocab_size,sizeof(float));
    s->ff_buf =(float*)calloc(cfg->ff_dim,sizeof(float));
    s->ff_buf2=(float*)calloc(cfg->ff_dim,sizeof(float));
    /* tmp must hold max(dim, ff_dim) floats */
    int tmp_sz = cfg->dim > cfg->ff_dim ? cfg->dim : cfg->ff_dim;
    s->tmp    =(float*)calloc(tmp_sz,sizeof(float));
    size_t kv_sz=(size_t)cfg->n_layers*cfg->max_seq_len*cfg->n_kv_heads*cfg->head_dim;
    s->k_cache=(float*)calloc(kv_sz,sizeof(float));
    s->v_cache=(float*)calloc(kv_sz,sizeof(float));
    if(!s->x||!s->xb||!s->q||!s->k||!s->v||!s->att||!s->logits
       ||!s->ff_buf||!s->ff_buf2||!s->k_cache||!s->v_cache||!s->tmp){
        fprintf(stderr,"OOM run state\n"); return -1;
    }
    size_t kv_mb = kv_sz * sizeof(float) / 1024 / 1024;
    printf("[run_state] KV cache: %zu MB\n", kv_mb);
    return 0;
}

static void free_run_state(RunState *s){
    free(s->x);free(s->xb);free(s->q);free(s->k);free(s->v);
    free(s->att);free(s->logits);free(s->ff_buf);free(s->ff_buf2);
    free(s->k_cache);free(s->v_cache);free(s->tmp);
}

/* ══════════════════════════ forward pass ══════════════════════════════ */

static float* forward(Weights *w,Config *cfg,RunState *s,int token,int pos){
    int dim=cfg->dim, kv_dim=cfg->n_kv_heads*cfg->head_dim, q_dim=cfg->n_heads*cfg->head_dim;
    float scale=1.0f/sqrtf((float)cfg->head_dim);

    float *emb=w->embed_tokens+(size_t)token*dim;
    memcpy(s->x,emb,dim*sizeof(float));

    for(int l=0;l<cfg->n_layers;l++){
        rmsnorm(s->xb,s->x,w->attn_norm[l],dim);

        memset(s->q,0,q_dim*sizeof(float));
        memset(s->k,0,kv_dim*sizeof(float));
        memset(s->v,0,kv_dim*sizeof(float));
        matmul_acc(s->q,s->xb,w->attn_q[l],q_dim, dim);
        matmul_acc(s->k,s->xb,w->attn_k[l],kv_dim,dim);
        matmul_acc(s->v,s->xb,w->attn_v[l],kv_dim,dim);

        rope(s->q,s->k,cfg->head_dim,cfg->n_heads,cfg->n_kv_heads,pos,cfg->rope_theta);

        size_t kv_off=((size_t)l*cfg->max_seq_len+pos)*kv_dim;
        memcpy(s->k_cache+kv_off,s->k,kv_dim*sizeof(float));
        memcpy(s->v_cache+kv_off,s->v,kv_dim*sizeof(float));

        memset(s->xb,0,dim*sizeof(float));
        for(int h=0;h<cfg->n_heads;h++){
            int kv_head=h/cfg->gqa_factor;
            float *qh=s->q+h*cfg->head_dim, *atth=s->att+h*cfg->max_seq_len;
            for(int t=0;t<=pos;t++){
                size_t kv_t=((size_t)l*cfg->max_seq_len+t)*kv_dim+(size_t)kv_head*cfg->head_dim;
                float *kt=s->k_cache+kv_t, dot=0.0f;
                for(int i=0;i<cfg->head_dim;i++) dot+=qh[i]*kt[i];
                atth[t]=dot*scale;
            }
            softmax(atth,pos+1);
            float *out_h=s->xb+h*cfg->head_dim;
            for(int t=0;t<=pos;t++){
                size_t kv_t=((size_t)l*cfg->max_seq_len+t)*kv_dim+(size_t)kv_head*cfg->head_dim;
                float *vt=s->v_cache+kv_t; float a=atth[t];
                for(int i=0;i<cfg->head_dim;i++) out_h[i]+=a*vt[i];
            }
        }

        /* output projection + residual — use s->tmp instead of VLA */
        matmul(s->tmp,s->xb,w->attn_o[l],dim,q_dim);
        for(int i=0;i<dim;i++) s->x[i]+=s->tmp[i];

        /* FFN */
        rmsnorm(s->xb,s->x,w->ff_norm[l],dim);
        matmul(s->ff_buf, s->xb,w->ff_gate[l],cfg->ff_dim,dim);
        matmul(s->ff_buf2,s->xb,w->ff_up[l],  cfg->ff_dim,dim);
        silu(s->ff_buf,cfg->ff_dim);
        for(int i=0;i<cfg->ff_dim;i++) s->ff_buf[i]*=s->ff_buf2[i];
        matmul(s->tmp,s->ff_buf,w->ff_down[l],dim,cfg->ff_dim);
        for(int i=0;i<dim;i++) s->x[i]+=s->tmp[i];
    }

    rmsnorm(s->x,s->x,w->norm,dim);
    matmul(s->logits,s->x,w->lm_head,cfg->vocab_size,dim);
    return s->logits;
}

static float* forwar8(Weight8 *w, Config *cfg, RunState *s, int token, int pos) {
    int dim    = cfg->dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;
    int q_dim  = cfg->n_heads    * cfg->head_dim;
    float scale = 1.0f / sqrtf((float)cfg->head_dim);

    /* ── embedding lookup: dequant one FP8 row ── */
    const uint8_t *emb = w->embed_tokens + (size_t)token * dim;
    for (int i = 0; i < dim; i++) s->x[i] = g_fp8_lut[emb[i]];

    for (int l = 0; l < cfg->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->attn_norm[l], dim);   /* norm weights are float */

        memset(s->q, 0, q_dim  * sizeof(float));
        memset(s->k, 0, kv_dim * sizeof(float));
        memset(s->v, 0, kv_dim * sizeof(float));
        fp8_sgemv_acc(s->q, w->attn_q[l], s->xb, q_dim,  dim);
        fp8_sgemv_acc(s->k, w->attn_k[l], s->xb, kv_dim, dim);
        fp8_sgemv_acc(s->v, w->attn_v[l], s->xb, kv_dim, dim);

        rope(s->q, s->k, cfg->head_dim, cfg->n_heads, cfg->n_kv_heads,
             pos, cfg->rope_theta);

        size_t kv_off = ((size_t)l * cfg->max_seq_len + pos) * kv_dim;
        memcpy(s->k_cache + kv_off, s->k, kv_dim * sizeof(float));
        memcpy(s->v_cache + kv_off, s->v, kv_dim * sizeof(float));

        memset(s->xb, 0, dim * sizeof(float));
        for (int h = 0; h < cfg->n_heads; h++) {
            int kv_head = h / cfg->gqa_factor;
            float *qh   = s->q   + h * cfg->head_dim;
            float *atth = s->att + h * cfg->max_seq_len;

            for (int t = 0; t <= pos; t++) {
                size_t kv_t = ((size_t)l * cfg->max_seq_len + t) * kv_dim
                              + (size_t)kv_head * cfg->head_dim;
                float *kt = s->k_cache + kv_t, dot = 0.0f;
                for (int i = 0; i < cfg->head_dim; i++) dot += qh[i] * kt[i];
                atth[t] = dot * scale;
            }
            softmax(atth, pos + 1);

            float *out_h = s->xb + h * cfg->head_dim;
            for (int t = 0; t <= pos; t++) {
                size_t kv_t = ((size_t)l * cfg->max_seq_len + t) * kv_dim
                              + (size_t)kv_head * cfg->head_dim;
                float *vt = s->v_cache + kv_t, a = atth[t];
                for (int i = 0; i < cfg->head_dim; i++) out_h[i] += a * vt[i];
            }
        }

        fp8_sgemv(s->tmp, w->attn_o[l], s->xb, dim, q_dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->tmp[i];

        /* FFN */
        rmsnorm(s->xb, s->x, w->ff_norm[l], dim);
        fp8_sgemv(s->ff_buf,  w->ff_gate[l], s->xb, cfg->ff_dim, dim);
        fp8_sgemv(s->ff_buf2, w->ff_up[l],   s->xb, cfg->ff_dim, dim);
        silu(s->ff_buf, cfg->ff_dim);
        for (int i = 0; i < cfg->ff_dim; i++) s->ff_buf[i] *= s->ff_buf2[i];
        fp8_sgemv(s->tmp, w->ff_down[l], s->ff_buf, dim, cfg->ff_dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->tmp[i];
    }

    rmsnorm(s->x, s->x, w->norm, dim);
    fp8_sgemv(s->logits, w->lm_head, s->x, cfg->vocab_size, dim);
    return s->logits;
}
/* ══════════════════════════ sampling ══════════════════════════════════ */

static int sample_argmax(float *logits,int n){
    int best=0; for(int i=1;i<n;i++)if(logits[i]>logits[best])best=i; return best;
}
static int sample_temperature(float *logits,int n,float temp){
    for(int i=0;i<n;i++)logits[i]/=temp; softmax(logits,n);
    float r=(float)rand()/(float)RAND_MAX, cum=0.0f;
    for(int i=0;i<n;i++){cum+=logits[i];if(r<cum)return i;} return n-1;
}

static double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
}

/* ══════════════════════════ main ══════════════════════════════════════ */

int main(int argc,char *argv[]){
    if(argc<3){
        fprintf(stderr,
            "Usage: %s <model_dir> \"<prompt>\" [max_tokens] [temperature]\n"
            "\n"
            "model_dir must contain:\n"
            "  model.llmbin   (fast — convert with convert_weights.py)  ← preferred\n"
            "  model.safetensors  (slow JSON parse on startup)\n"
            "  tokenizer.json\n"
            "\n"
            "BLAS backend: %s\n",
            argv[0], BLAS_BACKEND_NAME);
        return 1;
    }

    const char *model_dir=argv[1], *prompt=argv[2];
    int   max_tokens = argc>3 ? atoi(argv[3]) : 200;
    float temperature= argc>4 ? atof(argv[4]) : 0.0f;
    srand((unsigned)time(NULL));

    printf("╔══════════════════════════════════════╗\n");
    printf("║  llama.c  │  BLAS: %-17s║\n", BLAS_BACKEND_NAME);
    printf("╚══════════════════════════════════════╝\n");

    /* ── Try .llmbin first, fall back to .safetensors ── */
    char path_buf[512];
    LLMBin    lb; memset(&lb,0,sizeof(lb));
    // SafeTensors st; memset(&st,0,sizeof(st));
    Weights   w;  memset(&w, 0,sizeof(w));
    Config    cfg;
    int use_llmbin = 0;

    /* candidates for llmbin */
    const char *llmbin_candidates[] = {
        "%s/model.llmbin",
        "%s/model-00001-of-00001.llmbin",
        "%s",
        NULL
    };
    for(int ci=0;llmbin_candidates[ci];ci++){
        snprintf(path_buf,sizeof(path_buf),llmbin_candidates[ci],model_dir);
        struct stat _sb; if(stat(path_buf,&_sb)!=0) continue;
        /* check extension */
        const char *ext=strrchr(path_buf,'.'); if(!ext||strcmp(ext,".llmbin")!=0) continue;
        if(llmbin_open(&lb,path_buf)==0){ use_llmbin=1; break; }
    }

    if(use_llmbin){
        cfg=llmbin_detect_config(&lb);
        printf("[config] dim=%d layers=%d heads=%d kv_heads=%d ff=%d vocab=%d\n",
               cfg.dim,cfg.n_layers,cfg.n_heads,cfg.n_kv_heads,cfg.ff_dim,cfg.vocab_size);
        if(load_weights_llmbin(&w,&cfg,&lb)<0) return 1;
    } else {
        /* fall back to safetensors */
        // const char *st_candidates[]={
        //     "%s/model.safetensors",
        //     "%s/model-00001-of-00001.safetensors",
        //     "%s/model-00001-of-00002.safetensors",
        //     "%s", NULL
        // };
        // int opened=0;
        // for(int ci=0;st_candidates[ci];ci++){
        //     snprintf(path_buf,sizeof(path_buf),st_candidates[ci],model_dir);
        //     struct stat _sb; if(stat(path_buf,&_sb)!=0) continue;
        //     if(st_open(&st,path_buf)==0){opened=1;break;}
        // }
        // if(!opened){fprintf(stderr,"Could not open model weights in: %s\n",model_dir);return 1;}
        // cfg=st_detect_config(&st);
        // printf("[config] dim=%d layers=%d heads=%d kv_heads=%d ff=%d vocab=%d\n",
        //        cfg.dim,cfg.n_layers,cfg.n_heads,cfg.n_kv_heads,cfg.ff_dim,cfg.vocab_size);
        // if(load_weights_st(&w,&cfg,&st)<0) return 1;
        // st_close(&st);
    }

    /* Tokenizer */
    char tok_path[512]; snprintf(tok_path,sizeof(tok_path),"%s/tokenizer.json",model_dir);
    Tokenizer tok; memset(&tok,0,sizeof(tok));
    if(tok_load(&tok,tok_path)<0) return 1;

    /* Run state */
    RunState s; memset(&s,0,sizeof(s));
    if(alloc_run_state(&s,&cfg)<0) return 1;

    /* Encode */
    int ids[4096]; ids[0]=tok.bos;
    int n_prompt=tok_encode(&tok,prompt,ids+1,4095)+1;
    printf("[encode] Prompt tokens: %d\n",n_prompt);
    printf("\n--- Output ---\n"); fwrite(prompt,1,strlen(prompt),stdout); fflush(stdout);

    /* Inference */
    double t0=now_sec(), t_first=0.0; int n_gen=0, next=0;
    int total=n_prompt+max_tokens; if(total>cfg.max_seq_len) total=cfg.max_seq_len;

    for(int pos=0;pos<total;pos++){
        int token=(pos<n_prompt)?ids[pos]:next;
        float *logits=forward(&w,&cfg,&s,token,pos);
        if(pos==n_prompt-1) t_first=now_sec();
        if(pos<n_prompt-1) continue;

        next=(temperature<=0.0f)?sample_argmax(logits,cfg.vocab_size)
                                 :sample_temperature(logits,cfg.vocab_size,temperature);
        if(next==tok.eos||next==tok.bos){printf("\n[EOS]\n");break;}
        if(tok.vocab[next]&&tok.vocab_lens[next]>0){
            fwrite(tok.vocab[next],1,tok.vocab_lens[next],stdout); fflush(stdout);
        }
        n_gen++;
    }

    double t_end=now_sec();
    double t_pre=t_first-t0, t_gen=t_end-t_first;
    double tps=(t_gen>0&&n_gen>0)?(double)n_gen/t_gen:0.0;

    printf("\n\n══════════════════════════════════════\n");
    printf("  Backend       : %s\n",  BLAS_BACKEND_NAME);
    printf("  Weight format : %s\n",  use_llmbin?"llmbin (fast)":"safetensors");
    printf("  Model         : %dB\n", cfg.dim>=4096?8:1);
    printf("  Prompt tokens : %d\n",  n_prompt);
    printf("  Generated     : %d tokens\n",n_gen);
    printf("  Prefill time  : %.2f s (%.1f tok/s)\n",t_pre,t_pre>0?(double)n_prompt/t_pre:0.0);
    printf("  Generate time : %.2f s\n",t_gen);
    printf("  Throughput    : %.2f tok/s\n",tps);
    printf("══════════════════════════════════════\n");

    free_run_state(&s);
    tok_free(&tok);
    if(use_llmbin) llmbin_close(&lb);
    free_weights(&w,&cfg);
    return 0;
}