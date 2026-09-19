#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>

/*
    FAST BYTE-LEVEL BPE TOKENIZER TRAINER (v2)

    What changed vs the original and why:

    1. OCCURRENCE LISTS PER PAIR (the big one).
       The original scanned the ENTIRE linked-list sequence on every
       single merge to find occurrences of the winning pair, and
       refreshed the heap by scanning the ENTIRE hash table capacity
       every iteration. Both of those are O(corpus) / O(capacity) per
       merge, so total training cost was O(merges * (corpus + capacity))
       -- and capacity only ever grew, since the old hash_sub() never
       deleted anything. That is why the original visibly decays from
       fast to a crawl as training progresses (watch "pairs=" grow
       every single line in its output -- that number should almost
       never grow that large).

       Here every pair keeps an explicit, intrusive doubly-linked list
       of the sequence positions where it currently occurs. A merge
       only ever touches:
         - the occurrence list of the pair being merged (its own
           frequency worth of positions), and
         - the immediate left/right neighbors of each occurrence.
       Nothing else in the corpus is touched. This makes each merge
       cost proportional to how often that pair occurs, not to corpus
       size or hash table size.

    2. REAL HASH DELETION.
       hash_del() now does proper open-addressing deletion with
       backward-shift (no tombstones, no permanent growth), so the
       table stays proportional to the number of *distinct pairs that
       currently occur* -- which shrinks as merges consume pairs.

    3. INCREMENTAL HEAP, NO FULL-TABLE RESCANS.
       Only the O(1)-ish handful of pairs whose counts actually changed
       this iteration get pushed back onto the heap. The heap uses the
       standard lazy-deletion trick (an entry is stale if its count no
       longer matches the hash table) instead of ever being rebuilt
       from a full table scan.

    4. PARALLEL PREPROCESSING.
       The only truly parallel part of BPE training is the initial
       pass over the raw corpus: computing initial adjacent-pair counts
       and building initial occurrence lists. This is now split across
       `threads` worker threads over contiguous chunks of the input,
       with per-thread local hash tables merged at the end. The merge
       loop itself is inherently sequential (merge i+1 depends on the
       result of merge i), so it is not and cannot be parallelized --
       any tool claiming to multithread that loop is not doing real
       BPE.

    Build:
        gcc -O3 -march=native -flto -pthread tokenizer_fast.c -o tokenizer_fast

    Usage:
        ./tokenizer_fast data.txt tokenizer.bin 32768 8
*/

#define INITIAL_VOCAB 256
#define DEFAULT_VOCAB 32768

#define LOAD_FACTOR_NUM 7
#define LOAD_FACTOR_DEN 10
#define MIN_CAPACITY 64

#define MAGIC "BPE2"
#define VERSION 2

/* ============================================================
                           UTIL
   ============================================================ */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint64_t hash64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline size_t next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

/* ============================================================
                         TOKEN / PAIR
   ============================================================ */

typedef uint32_t token_t;
typedef uint64_t pair_t;

static inline pair_t make_pair(token_t a, token_t b) {
    return ((uint64_t)a << 32) | (uint64_t)b;
}
static inline token_t pair_left(pair_t p)  { return (token_t)(p >> 32); }
static inline token_t pair_right(pair_t p) { return (token_t)p; }

typedef struct {
    token_t left;
    token_t right;
} Token;

/* ============================================================
        HASH TABLE: pair -> PairInfo (count + occurrence list head)

    Real deletion via backward-shift open addressing, so the table
    shrinks (logically -- entries get removed) instead of growing
    forever like the original's tombstone-free "set to zero" approach.
   ============================================================ */

typedef struct {
    uint64_t count;      /* current occurrence count of this pair    */
    uint32_t occ_head;   /* index into the global Occ pool, or NONE  */
} PairInfo;

#define NONE UINT32_MAX

typedef struct {
    pair_t key;
    PairInfo info;
    uint8_t used;
} HashEntry;

typedef struct {
    HashEntry *entries;
    size_t capacity;   /* power of two */
    size_t size;       /* live entries */
} HashTable;

static void hash_init(HashTable *h, size_t capacity) {
    capacity = next_pow2(capacity < MIN_CAPACITY ? MIN_CAPACITY : capacity);
    h->capacity = capacity;
    h->size = 0;
    h->entries = calloc(capacity, sizeof(HashEntry));
    if (!h->entries) { perror("calloc"); exit(1); }
}

static void hash_free(HashTable *h) {
    free(h->entries);
    h->entries = NULL;
    h->capacity = 0;
    h->size = 0;
}

static inline size_t hash_index(const HashTable *h, pair_t key) {
    return hash64(key) & (h->capacity - 1);
}

static void hash_rehash(HashTable *h, size_t new_capacity) {
    size_t old_capacity = h->capacity;
    HashEntry *old_entries = h->entries;

    new_capacity = next_pow2(new_capacity < MIN_CAPACITY ? MIN_CAPACITY : new_capacity);
    h->capacity = new_capacity;
    h->size = 0;
    h->entries = calloc(new_capacity, sizeof(HashEntry));
    if (!h->entries) { perror("calloc"); exit(1); }

    for (size_t i = 0; i < old_capacity; i++) {
        if (!old_entries[i].used) continue;
        size_t idx = hash_index(h, old_entries[i].key);
        while (h->entries[idx].used) idx = (idx + 1) & (h->capacity - 1);
        h->entries[idx] = old_entries[i];
        h->size++;
    }
    free(old_entries);
}

static inline void hash_maybe_grow(HashTable *h) {
    if (h->size * LOAD_FACTOR_DEN >= h->capacity * LOAD_FACTOR_NUM) {
        hash_rehash(h, h->capacity * 2);
    }
}

/* Returns pointer to PairInfo, or NULL if absent. Pointer is only
   valid until the next insertion/deletion/rehash on this table. */
static PairInfo *hash_get(HashTable *h, pair_t key) {
    size_t idx = hash_index(h, key);
    while (1) {
        HashEntry *e = &h->entries[idx];
        if (!e->used) return NULL;
        if (e->key == key) return &e->info;
        idx = (idx + 1) & (h->capacity - 1);
    }
}

/* Insert or fetch-for-update. Always returns a valid pointer. */
static PairInfo *hash_get_or_insert(HashTable *h, pair_t key, PairInfo init) {
    hash_maybe_grow(h);
    size_t idx = hash_index(h, key);
    while (1) {
        HashEntry *e = &h->entries[idx];
        if (!e->used) {
            e->used = 1;
            e->key = key;
            e->info = init;
            h->size++;
            return &e->info;
        }
        if (e->key == key) return &e->info;
        idx = (idx + 1) & (h->capacity - 1);
    }
}

/* Real deletion with backward-shift so the table doesn't accumulate
   permanent garbage the way the original's zero-out approach did. */
static void hash_del(HashTable *h, pair_t key) {
    size_t idx = hash_index(h, key);
    while (1) {
        HashEntry *e = &h->entries[idx];
        if (!e->used) return; /* not present */
        if (e->key == key) break;
        idx = (idx + 1) & (h->capacity - 1);
    }

    size_t i = idx;
    h->entries[i].used = 0;
    h->size--;

    size_t j = i;
    while (1) {
        j = (j + 1) & (h->capacity - 1);
        if (!h->entries[j].used) break;

        size_t k = hash_index(h, h->entries[j].key);
        /* Standard backward-shift deletion condition. */
        int shift;
        if (i <= j) {
            shift = (k <= i || k > j);
        } else {
            shift = (k <= i && k > j);
        }
        if (shift) {
            h->entries[i] = h->entries[j];
            h->entries[j].used = 0;
            i = j;
        }
    }
}

/* ============================================================
                    OCCURRENCE LIST POOL

    One intrusive doubly-linked list per pair, threaded through a
    flat array indexed by sequence position. occ_prev[i]/occ_next[i]
    link position i within the occurrence list of whatever pair
    currently starts at i (i.e. token[i],token[next[i]]).
   ============================================================ */

typedef struct {
    uint32_t *occ_prev;
    uint32_t *occ_next;
} OccPool;

static void occ_pool_init(OccPool *o, size_t n) {
    o->occ_prev = malloc(n * sizeof(uint32_t));
    o->occ_next = malloc(n * sizeof(uint32_t));
    if (!o->occ_prev || !o->occ_next) { perror("malloc occ"); exit(1); }
}

static void occ_pool_free(OccPool *o) {
    free(o->occ_prev);
    free(o->occ_next);
}

/* Unlink position `pos` from whatever occurrence list it's currently
   in for hash table `h` under `key`, fixing up the head pointer. */
static void occ_remove(HashTable *h, OccPool *o, pair_t key, uint32_t pos) {
    PairInfo *info = hash_get(h, key);
    if (!info) return;

    uint32_t p = o->occ_prev[pos];
    uint32_t n = o->occ_next[pos];

    if (p != NONE) o->occ_next[p] = n;
    if (n != NONE) o->occ_prev[n] = p;
    if (info->occ_head == pos) info->occ_head = n;
}

/* Push `pos` onto the front of the occurrence list for `key`,
   creating the hash entry if needed, and bump its count. */
static void occ_add(HashTable *h, OccPool *o, pair_t key, uint32_t pos) {
    PairInfo init = { .count = 0, .occ_head = NONE };
    PairInfo *info = hash_get_or_insert(h, key, init);

    o->occ_prev[pos] = NONE;
    o->occ_next[pos] = info->occ_head;
    if (info->occ_head != NONE) o->occ_prev[info->occ_head] = pos;
    info->occ_head = pos;
    info->count++;
}

/* Decrement count for `key` and delete the hash entry entirely once
   it reaches zero, so the table never accumulates dead pairs. */
static void occ_dec_count(HashTable *h, pair_t key) {
    PairInfo *info = hash_get(h, key);
    if (!info) return;
    if (info->count <= 1) {
        hash_del(h, key);
    } else {
        info->count--;
    }
}

/* ============================================================
                         HEAP (max-heap on count)
   ============================================================ */

typedef struct { pair_t pair; uint64_t count; } HeapEntry;
typedef struct { HeapEntry *data; size_t size; size_t capacity; } Heap;

static int heap_better(HeapEntry a, HeapEntry b) {
    if (a.count != b.count) return a.count > b.count;
    return a.pair < b.pair;
}

static void heap_init(Heap *h, size_t capacity) {
    h->size = 0;
    h->capacity = capacity < 16 ? 16 : capacity;
    h->data = malloc(h->capacity * sizeof(HeapEntry));
    if (!h->data) { perror("malloc"); exit(1); }
}

static void heap_free(Heap *h) { free(h->data); }

static void heap_push(Heap *h, HeapEntry item) {
    if (h->size >= h->capacity) {
        h->capacity *= 2;
        h->data = realloc(h->data, h->capacity * sizeof(HeapEntry));
        if (!h->data) { perror("realloc"); exit(1); }
    }
    size_t i = h->size++;
    h->data[i] = item;
    while (i) {
        size_t parent = (i - 1) / 2;
        if (!heap_better(h->data[i], h->data[parent])) break;
        HeapEntry tmp = h->data[i]; h->data[i] = h->data[parent]; h->data[parent] = tmp;
        i = parent;
    }
}

static HeapEntry heap_pop(Heap *h) {
    HeapEntry root = h->data[0];
    h->size--;
    if (h->size == 0) return root;
    h->data[0] = h->data[h->size];
    size_t i = 0;
    while (1) {
        size_t left = i * 2 + 1, right = left + 1, best = i;
        if (left < h->size && heap_better(h->data[left], h->data[best])) best = left;
        if (right < h->size && heap_better(h->data[right], h->data[best])) best = right;
        if (best == i) break;
        HeapEntry tmp = h->data[i]; h->data[i] = h->data[best]; h->data[best] = tmp;
        i = best;
    }
    return root;
}

/* ============================================================
                          SEQUENCE
   ============================================================ */

typedef struct {
    uint32_t *token;
    uint32_t *prev;
    uint32_t *next;
    size_t length;   /* fixed allocation length            */
    uint32_t head;   /* index of first live position, or NONE */
} Sequence;

static void sequence_init(Sequence *s, const uint8_t *data, size_t length) {
    s->length = length;
    s->token = malloc(length * sizeof(uint32_t));
    s->prev  = malloc(length * sizeof(uint32_t));
    s->next  = malloc(length * sizeof(uint32_t));
    if (!s->token || !s->prev || !s->next) { perror("malloc sequence"); exit(1); }

    for (size_t i = 0; i < length; i++) {
        s->token[i] = data[i];
        s->prev[i]  = i == 0 ? NONE : (uint32_t)(i - 1);
        s->next[i]  = (i + 1 >= length) ? NONE : (uint32_t)(i + 1);
    }
    s->head = length > 0 ? 0 : NONE;
}

static void sequence_free(Sequence *s) {
    free(s->token);
    free(s->prev);
    free(s->next);
}

/* ============================================================
              PARALLEL INITIAL PAIR COUNTING

    Each thread scans a contiguous chunk of the corpus, building a
    local hash table of pair counts and (single-threaded-equivalent)
    occurrence positions. Local tables are then merged into the global
    one. Threads operate on disjoint index ranges, and each thread's
    chunk boundary is handled by having the *next* chunk's thread own
    the boundary pair (chunk i counts pairs (i, i+1) for all i in its
    range except the very last position of the chunk, which is instead
    counted by treating ranges as overlapping by one token). This
    avoids any cross-thread synchronization during the scan itself.
   ============================================================ */

typedef struct {
    const uint8_t *data;
    size_t start;   /* inclusive */
    size_t end;     /* exclusive, but this thread also reads data[end]
                        if it exists, to count the boundary pair */
    size_t total_length;
    HashTable local; /* pair -> PairInfo{count, occ_head=NONE} (occ
                         lists aren't needed here; positions are
                         re-scanned into the real OccPool afterward) */
} InitWorker;

static void *init_worker_fn(void *arg) {
    InitWorker *w = (InitWorker *)arg;
    hash_init(&w->local, 1024);

    size_t end = w->end;
    if (end < w->total_length) end += 1; /* include boundary pair */

    for (size_t i = w->start; i + 1 < end && i + 1 < w->total_length; i++) {
        pair_t p = make_pair(w->data[i], w->data[i + 1]);
        PairInfo init = { .count = 0, .occ_head = NONE };
        PairInfo *info = hash_get_or_insert(&w->local, p, init);
        info->count++;
    }
    return NULL;
}

/* ============================================================
                     TRAINING STRUCTURE
   ============================================================ */

typedef struct {
    Token *vocab;
    size_t vocab_size;
    size_t vocab_capacity;

    pair_t *merges;
    size_t merge_count;
    size_t merge_capacity;

    HashTable pairs;   /* pair -> {count, occ_head} */
    OccPool occ;
    Heap heap;
    Sequence seq;
} Trainer;

static token_t add_token(Trainer *t, token_t left, token_t right) {
    if (t->vocab_size >= t->vocab_capacity) {
        t->vocab_capacity *= 2;
        t->vocab = realloc(t->vocab, t->vocab_capacity * sizeof(Token));
        if (!t->vocab) { perror("realloc vocab"); exit(1); }
    }
    token_t id = (token_t)t->vocab_size;
    t->vocab[id].left = left;
    t->vocab[id].right = right;
    t->vocab_size++;
    return id;
}

/* ============================================================
          BUILD INITIAL PAIR COUNTS + OCCURRENCE LISTS
          (parallel counting, then a single-threaded pass to
           actually thread the occurrence lists, since those
           lists are inherently a single shared structure)
   ============================================================ */

static void build_initial_state(Trainer *t, const uint8_t *data, int threads) {
    fprintf(stderr, "Building initial pair counts (%d threads)...\n", threads);
    uint64_t t0 = now_ns();

    size_t n = t->seq.length;
    if (threads < 1) threads = 1;
    if ((size_t)threads > n) threads = n > 0 ? (int)n : 1;

    InitWorker *workers = calloc(threads, sizeof(InitWorker));
    pthread_t *tids = calloc(threads, sizeof(pthread_t));

    size_t chunk = (n + threads - 1) / threads;
    for (int i = 0; i < threads; i++) {
        workers[i].data = data;
        workers[i].start = (size_t)i * chunk;
        workers[i].end = workers[i].start + chunk;
        if (workers[i].end > n) workers[i].end = n;
        workers[i].total_length = n;
    }

    for (int i = 0; i < threads; i++) {
        pthread_create(&tids[i], NULL, init_worker_fn, &workers[i]);
    }
    for (int i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
    }

    /* Merge per-thread local counts into a size estimate so we can
       size the global table well up front (avoids rehash storms). */
    size_t distinct_estimate = 0;
    for (int i = 0; i < threads; i++) distinct_estimate += workers[i].local.size;
    hash_init(&t->pairs, distinct_estimate * 2 + 64);

    for (int i = 0; i < threads; i++) {
        HashTable *loc = &workers[i].local;
        for (size_t j = 0; j < loc->capacity; j++) {
            if (!loc->entries[j].used) continue;
            PairInfo init = { .count = 0, .occ_head = NONE };
            PairInfo *g = hash_get_or_insert(&t->pairs, loc->entries[j].key, init);
            g->count += loc->entries[j].info.count;
        }
        hash_free(loc);
    }
    free(workers);
    free(tids);

    fprintf(stderr, "  counting: %.3fs, unique pairs: %zu\n",
            (double)(now_ns() - t0) / 1e9, t->pairs.size);

    /* Occurrence lists are a single shared linked structure, so build
       them with one sequential pass (this is a simple O(n) scan, not
       the O(capacity)-per-merge cost the original paid repeatedly). */
    t0 = now_ns();
    occ_pool_init(&t->occ, n);
    for (size_t i = 0; i + 1 < n; i++) {
        pair_t p = make_pair(t->seq.token[i], t->seq.token[i + 1]);
        PairInfo *info = hash_get(&t->pairs, p);
        /* info must exist: it was counted above */
        t->occ.occ_prev[i] = NONE;
        t->occ.occ_next[i] = info->occ_head;
        if (info->occ_head != NONE) t->occ.occ_prev[info->occ_head] = i;
        info->occ_head = i;
    }
    fprintf(stderr, "  occurrence lists: %.3fs\n", (double)(now_ns() - t0) / 1e9);
}

static void build_heap(Trainer *t) {
    heap_init(&t->heap, t->pairs.size + 1024);
    for (size_t i = 0; i < t->pairs.capacity; i++) {
        HashEntry *e = &t->pairs.entries[i];
        if (!e->used || e->info.count == 0) continue;
        heap_push(&t->heap, (HeapEntry){ e->key, e->info.count });
    }
}

/* ============================================================
                MERGE (occurrence-list driven)

    Only touches positions in the winning pair's occurrence list and
    their immediate neighbors. This replaces the original's full
    linked-list scan of the entire corpus per merge.
   ============================================================ */

static uint64_t merge_pair(Trainer *t, pair_t target, token_t new_token) {
    token_t a = pair_left(target);
    token_t b = pair_right(target);
    uint64_t frequency = 0;

    PairInfo *target_info = hash_get(&t->pairs, target);
    uint32_t head = target_info ? target_info->occ_head : NONE;

    uint32_t pos = head;
    while (pos != NONE) {
        uint32_t next_occ = t->occ.occ_next[pos]; /* save before mutation */

        uint32_t i = pos;
        uint32_t n = t->seq.next[i];

        /* Defensive check: with correct bookkeeping this always holds. */
        if (n != NONE && t->seq.token[i] == a && t->seq.token[n] == b) {
            uint32_t p  = t->seq.prev[i];
            uint32_t nn = t->seq.next[n];

            /* Remove the two neighbor pairs that touch this occurrence,
               both from the hash counts and from their occurrence lists. */
            if (p != NONE) {
                pair_t old_left = make_pair(t->seq.token[p], a);
                occ_remove(&t->pairs, &t->occ, old_left, p);
                occ_dec_count(&t->pairs, old_left);
            }
            if (nn != NONE) {
                pair_t old_right = make_pair(b, t->seq.token[nn]);
                occ_remove(&t->pairs, &t->occ, old_right, n);
                occ_dec_count(&t->pairs, old_right);
            }

            /* Splice out node n, i becomes the merged token. */
            t->seq.token[i] = new_token;
            t->seq.next[i] = nn;
            if (nn != NONE) t->seq.prev[nn] = i;

            /* Add the two new neighbor pairs created by the merge. */
            if (p != NONE) {
                pair_t new_left = make_pair(t->seq.token[p], new_token);
                occ_add(&t->pairs, &t->occ, new_left, p);
            }
            if (nn != NONE) {
                pair_t new_right = make_pair(new_token, t->seq.token[nn]);
                occ_add(&t->pairs, &t->occ, new_right, i);
            }

            frequency++;
        }

        pos = next_occ;
    }

    /* The target pair itself is now fully consumed. */
    hash_del(&t->pairs, target);

    return frequency;
}

/* ============================================================
                       TRAIN

    After each merge, only the handful of pairs whose counts actually
    changed (the neighbor pairs touched above) get pushed back onto
    the heap -- no full-table rescans. We track "dirty" pairs in a
    small growable buffer during merge_pair via a thread-local-ish
    static isn't safe for reentrancy, so instead we simply push the
    known-changed pairs right here in train(), since merge_pair
    already knows exactly which pairs changed.
   ============================================================ */

typedef struct {
    pair_t *items;
    size_t count;
    size_t capacity;
} PairSet;

static void pairset_init(PairSet *s) {
    s->capacity = 64;
    s->count = 0;
    s->items = malloc(s->capacity * sizeof(pair_t));
}
static void pairset_push(PairSet *s, pair_t p) {
    if (s->count >= s->capacity) {
        s->capacity *= 2;
        s->items = realloc(s->items, s->capacity * sizeof(pair_t));
    }
    s->items[s->count++] = p;
}
static void pairset_clear(PairSet *s) { s->count = 0; }
static void pairset_free(PairSet *s) { free(s->items); }

/* Re-implementation of merge_pair that also records which pairs were
   touched, so train() can push exactly those (and only those) back
   onto the heap. Functionally identical to merge_pair above. */
static uint64_t merge_pair_tracked(Trainer *t, pair_t target, token_t new_token, PairSet *dirty) {
    token_t a = pair_left(target);
    token_t b = pair_right(target);
    uint64_t frequency = 0;

    PairInfo *target_info = hash_get(&t->pairs, target);
    uint32_t head = target_info ? target_info->occ_head : NONE;

    uint32_t pos = head;
    while (pos != NONE) {
        uint32_t next_occ = t->occ.occ_next[pos];

        uint32_t i = pos;
        uint32_t n = t->seq.next[i];

        if (n != NONE && t->seq.token[i] == a && t->seq.token[n] == b) {
            uint32_t p  = t->seq.prev[i];
            uint32_t nn = t->seq.next[n];

            if (p != NONE) {
                pair_t old_left = make_pair(t->seq.token[p], a);
                occ_remove(&t->pairs, &t->occ, old_left, p);
                occ_dec_count(&t->pairs, old_left);
                pairset_push(dirty, old_left);
            }
            if (nn != NONE) {
                pair_t old_right = make_pair(b, t->seq.token[nn]);
                occ_remove(&t->pairs, &t->occ, old_right, n);
                occ_dec_count(&t->pairs, old_right);
                pairset_push(dirty, old_right);
            }

            t->seq.token[i] = new_token;
            t->seq.next[i] = nn;
            if (nn != NONE) t->seq.prev[nn] = i;

            if (p != NONE) {
                pair_t new_left = make_pair(t->seq.token[p], new_token);
                occ_add(&t->pairs, &t->occ, new_left, p);
                pairset_push(dirty, new_left);
            }
            if (nn != NONE) {
                pair_t new_right = make_pair(new_token, t->seq.token[nn]);
                occ_add(&t->pairs, &t->occ, new_right, i);
                pairset_push(dirty, new_right);
            }

            frequency++;
        }

        pos = next_occ;
    }

    hash_del(&t->pairs, target);
    return frequency;
}

static int pair_cmp(const void *a, const void *b) {
    pair_t pa = *(const pair_t *)a;
    pair_t pb = *(const pair_t *)b;
    if (pa < pb) return -1;
    if (pa > pb) return 1;
    return 0;
}

static void train(Trainer *t, size_t target_vocab) {
    size_t merges = target_vocab - INITIAL_VOCAB;
    uint64_t start = now_ns();

    PairSet dirty;
    pairset_init(&dirty);

    for (size_t iteration = 0; iteration < merges; iteration++) {
        HeapEntry best;
        int found = 0;

        while (t->heap.size) {
            best = heap_pop(&t->heap);
            PairInfo *info = hash_get(&t->pairs, best.pair);
            uint64_t current = info ? info->count : 0;
            /* Lazy heap deletion: skip stale entries. */
            if (current == best.count && current > 0) { found = 1; break; }
        }

        if (!found) {
            fprintf(stderr, "No more mergeable pairs.\n");
            break;
        }

        token_t a = pair_left(best.pair);
        token_t b = pair_right(best.pair);
        token_t new_token = add_token(t, a, b);

        if (t->merge_count >= t->merge_capacity) {
            t->merge_capacity *= 2;
            t->merges = realloc(t->merges, t->merge_capacity * sizeof(pair_t));
            if (!t->merges) { perror("realloc merges"); exit(1); }
        }
        t->merges[t->merge_count++] = best.pair;

        pairset_clear(&dirty);
        uint64_t frequency = merge_pair_tracked(t, best.pair, new_token, &dirty);

        /* Dedupe before pushing: merge_pair_tracked records one dirty entry per
        OCCURRENCE touched, but most occurrences share the same neighboring
        pair. Pushing each occurrence separately makes the heap grow with
        frequency (millions) instead of with distinct pairs (hundreds). */
        if (dirty.count > 1) {
            qsort(dirty.items, dirty.count, sizeof(pair_t), pair_cmp);
        }
        for (size_t i = 0; i < dirty.count; i++) {
            if (i > 0 && dirty.items[i] == dirty.items[i - 1]) continue;
            PairInfo *info = hash_get(&t->pairs, dirty.items[i]);
            if (info && info->count > 0) {
                heap_push(&t->heap, (HeapEntry){ dirty.items[i], info->count });
            }
        }

        if (iteration % 200 == 0 || iteration + 1 == merges) {
            double elapsed = (double)(now_ns() - start) / 1e9;
            fprintf(stderr,
                "\rmerge %7zu/%7zu token=%7u freq=%10llu live_pairs=%8zu heap=%8zu time=%.1fs",
                iteration + 1, merges, new_token,
                (unsigned long long)frequency, t->pairs.size, t->heap.size, elapsed);
            fflush(stderr);
        }
    }

    pairset_free(&dirty);
    fprintf(stderr, "\n");
}

/* ============================================================
                          FILE FORMAT
   ============================================================ */

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t merge_count;
    uint32_t reserved;
} FileHeader;

static int save_tokenizer(Trainer *t, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); return -1; }

    FileHeader header;
    memcpy(header.magic, MAGIC, 4);
    header.version = VERSION;
    header.vocab_size = (uint32_t)t->vocab_size;
    header.merge_count = (uint32_t)t->merge_count;
    header.reserved = 0;

    fwrite(&header, sizeof(header), 1, f);
    fwrite(t->vocab, sizeof(Token), t->vocab_size, f);
    fwrite(t->merges, sizeof(pair_t), t->merge_count, f);
    fclose(f);
    return 0;
}

/* ============================================================
                          CLEANUP
   ============================================================ */

static void trainer_free(Trainer *t) {
    free(t->vocab);
    free(t->merges);
    hash_free(&t->pairs);
    occ_pool_free(&t->occ);
    heap_free(&t->heap);
    sequence_free(&t->seq);
}

/* ============================================================
                             MAIN
   ============================================================ */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "\nUsage:\n  %s <dataset> <output.bin> [vocab_size] [threads]\n\n"
            "Example:\n  %s data.txt tokenizer.bin 32768 8\n\n",
            argv[0], argv[0]);
        return 1;
    }

    const char *dataset = argv[1];
    const char *output = argv[2];
    size_t vocab_size = argc >= 4 ? strtoull(argv[3], NULL, 10) : DEFAULT_VOCAB;
    int threads = argc >= 5 ? atoi(argv[4]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (threads < 1) threads = 1;

    printf("========================================\n"
           " Fast BPE tokenizer trainer (v2)\n"
           "========================================\n");
    printf("Dataset : %s\nOutput  : %s\nVocab   : %zu\nThreads : %d\n\n",
           dataset, output, vocab_size, threads);

    if (vocab_size <= INITIAL_VOCAB) {
        fprintf(stderr, "vocab_size must be > %d\n", INITIAL_VOCAB);
        return 1;
    }

    int fd = open(dataset, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return 1; }
    size_t file_size = (size_t)st.st_size;

    printf("Dataset size: %.2f MB\n", (double)file_size / (1024.0 * 1024.0));

    if (file_size == 0) {
        fprintf(stderr, "empty dataset\n");
        close(fd);
        return 1;
    }

    uint64_t t0 = now_ns();
    uint8_t *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    close(fd);
    printf("mmap: %.3fs\n", (double)(now_ns() - t0) / 1e9);

    madvise(data, file_size, MADV_SEQUENTIAL);

    Trainer t;
    memset(&t, 0, sizeof(t));

    t.vocab_capacity = vocab_size + 16;
    t.vocab = malloc(t.vocab_capacity * sizeof(Token));
    t.merge_capacity = vocab_size;
    t.merges = malloc(t.merge_capacity * sizeof(pair_t));
    if (!t.vocab || !t.merges) { perror("malloc"); munmap(data, file_size); return 1; }

    for (size_t i = 0; i < INITIAL_VOCAB; i++) {
        t.vocab[i].left = (uint32_t)i;
        t.vocab[i].right = 0;
    }
    t.vocab_size = INITIAL_VOCAB;

    t0 = now_ns();
    sequence_init(&t.seq, data, file_size);
    printf("Sequence: %.3fs\n", (double)(now_ns() - t0) / 1e9);

    t0 = now_ns();
    build_initial_state(&t, data, threads);
    printf("Pair statistics total: %.3fs\n", (double)(now_ns() - t0) / 1e9);

    munmap(data, file_size);

    t0 = now_ns();
    build_heap(&t);
    printf("Heap: %.3fs\n", (double)(now_ns() - t0) / 1e9);

    t0 = now_ns();
    train(&t, vocab_size);
    printf("Training: %.3fs\n", (double)(now_ns() - t0) / 1e9);

    t0 = now_ns();
    if (save_tokenizer(&t, output) != 0) {
        trainer_free(&t);
        return 1;
    }
    printf("Saved tokenizer: %s\n", output);
    printf("Save: %.3fs\n", (double)(now_ns() - t0) / 1e9);

    printf("\nFinal vocabulary: %zu\n", t.vocab_size);
    printf("Total merges: %zu\n", t.merge_count);

    trainer_free(&t);
    return 0;
}