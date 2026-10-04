#include "uni_tables.h"
#include "pretok.h"
#include <stdlib.h>
#include <string.h>

static uint8_t cp_flags(uint32_t cp) {
    if (cp < 128) return ASCII_FLAGS[cp];
    size_t lo = 0, hi = UNI_NRANGES;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < UNI_RANGES[mid].lo) hi = mid;
        else if (cp > UNI_RANGES[mid].hi) lo = mid + 1;
        else return UNI_RANGES[mid].f;
    }
    return 0;
}

static size_t next_cp(const unsigned char *s, size_t n, size_t i, uint32_t *cp) {
    unsigned char b = s[i];
    if (b < 0x80) { *cp = b; return 1; }
    size_t need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 0;
    if (need == 0 || i + need > n) { *cp = 0xFFFD; return 1; }
    uint32_t c = b & (0xFF >> (need + 1));
    for (size_t k = 1; k < need; k++) {
        if ((s[i + k] & 0xC0) != 0x80) { *cp = 0xFFFD; return 1; }
        c = (c << 6) | (s[i + k] & 0x3F);
    }
    *cp = c;
    return need;
}

static size_t at(const Src *t, size_t i, uint32_t *cp) {
    if (i >= t->n) { *cp = 0xFFFFFFFF; return 0; }
    return next_cp(t->s, t->n, i, cp);
}
static uint8_t fl(uint32_t cp) { return cp == 0xFFFFFFFF ? 0 : cp_flags(cp); }

static size_t match_contraction(const Src *t, size_t i) {
    const unsigned char *s = t->s;
    if (i + 1 >= t->n || s[i] != '\'') return 0;
    unsigned char a = s[i + 1] | 0x20;
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (i + 2 < t->n) {
        unsigned char b = s[i + 2] | 0x20;
        if ((a == 'r' || a == 'v') && b == 'e') return 3;
        if (a == 'l' && b == 'l') return 3;
    }
    return 0;
}

static size_t match_word(const Src *t, size_t p, int alt) {
    uint32_t cp; size_t len, i = p;
    if (alt == 1) {
        size_t last_w = (size_t)-1;
        for (;;) {
            len = at(t, i, &cp);
            uint8_t f = fl(cp);
            if (f & F_W) last_w = i;
            if (len == 0 || !(f & F_U)) break;
            i += len;
        }
        if (last_w == (size_t)-1) return 0;
        i = last_w;
        for (;;) {
            len = at(t, i, &cp);
            if (len == 0 || !(fl(cp) & F_W)) break;
            i += len;
        }
    } else {
        for (;;) {
            len = at(t, i, &cp);
            if (len == 0 || !(fl(cp) & F_U)) break;
            i += len;
        }
        if (i == p) return 0;
        for (;;) {
            len = at(t, i, &cp);
            if (len == 0 || !(fl(cp) & F_W)) break;
            i += len;
        }
    }
    return i + match_contraction(t, i);
}

static size_t try_words(const Src *t, size_t i) {
    uint32_t cp; size_t len = at(t, i, &cp), e;
    if (len == 0) return 0;
    int has_prefix = cp != '\r' && cp != '\n' && !(fl(cp) & (F_L | F_N));
    for (int alt = 1; alt <= 2; alt++) {
        if (has_prefix && (e = match_word(t, i + len, alt))) return e;
        if ((e = match_word(t, i, alt))) return e;
    }
    return 0;
}

static size_t match_digits(const Src *t, size_t i) {
    uint32_t cp; size_t j = i, len; int k = 0;
    while (k < 3) {
        len = at(t, j, &cp);
        if (len == 0 || !(fl(cp) & F_N)) break;
        j += len; k++;
    }
    return k ? j : 0;
}

static size_t match_punct(const Src *t, size_t i) {
    uint32_t cp; size_t j = i, len = at(t, j, &cp);
    if (len && cp == ' ') j += len;
    size_t k = j; int cnt = 0;
    for (;;) {
        len = at(t, k, &cp);
        if (len == 0 || (fl(cp) & (F_S | F_L | F_N))) break;
        k += len; cnt++;
    }
    if (!cnt) return 0;
    for (;;) {
        len = at(t, k, &cp);
        if (len == 0 || !(cp == '\r' || cp == '\n' || cp == '/')) break;
        k += len;
    }
    return k;
}

static size_t match_ws(const Src *t, size_t i) {
    uint32_t cp; size_t j = i, len, last_nl_end = 0, last_start = i, count = 0;
    for (;;) {
        len = at(t, j, &cp);
        if (len == 0 || !(fl(cp) & F_S)) break;
        last_start = j; j += len; count++;
        if (cp == '\r' || cp == '\n') last_nl_end = j;
    }
    if (!count) return 0;
    if (last_nl_end) return last_nl_end;
    if (j >= t->n) return j;
    if (count >= 2) return last_start;
    return j;
}

size_t pretok_next(const Src *t, size_t i) {
    size_t e;
    if ((e = try_words(t, i)))  return e;
    if ((e = match_digits(t, i))) return e;
    if ((e = match_punct(t, i)))  return e;
    if ((e = match_ws(t, i)))     return e;
    uint32_t cp;
    return i + next_cp(t->s, t->n, i, &cp);
}

Chunks pretok_split(const char *text, size_t n) {
    Chunks c = { NULL, NULL, 0 };
    size_t cap = 1024;
    c.items = malloc(cap * sizeof(char *));
    c.lens  = malloc(cap * sizeof(size_t));
    Src t = { (const unsigned char *)text, n };
    for (size_t i = 0; i < n; ) {
        size_t e = pretok_next(&t, i);
        if (c.count == cap) {
            cap *= 2;
            c.items = realloc(c.items, cap * sizeof(char *));
            c.lens  = realloc(c.lens,  cap * sizeof(size_t));
        }
        size_t len = e - i;
        char *s = malloc(len + 1);
        memcpy(s, text + i, len);
        s[len] = '\0';
        c.items[c.count] = s;
        c.lens[c.count]  = len;
        c.count++;
        i = e;
    }
    return c;
}

void chunks_free(Chunks *c) {
    for (size_t k = 0; k < c->count; k++) free(c->items[k]);
    free(c->items); free(c->lens);
    c->items = NULL; c->lens = NULL; c->count = 0;
}
