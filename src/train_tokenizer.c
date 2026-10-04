#include "pretok.h"
#include "types.h"
#include "utils.c"
#include "arena.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_VOCAB_LEN 2048

char *read_file(Arena *arena, const char *path, size_t *size) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return NULL;

  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);
  rewind(fp);

  char *buffer = (char *)arena_alloc(arena, *size + 1);
  if (!buffer) { fclose(fp); return NULL; }

  size_t read = fread(buffer, 1, *size, fp);
  buffer[read] = '\0';

  fclose(fp);
  return buffer;
}

int main(void){
  Arena arena = arena_create(GiB(1));
  HashTable *vocab = create_hash_table(&arena, 1000);
  HashTable *r_vocab = create_hash_table(&arena, 1000);
  HashTable *merges = create_hash_table(&arena, 1000);
  
  size_t f_size;
  char *contents = read_file(&arena, "data.txt", &f_size);

  if (!contents) {
      perror("read_file");
      return 1;
  }

  // loading vocab with initial 256 chars
  for (uint32 i = 0; i < 256; i++){
    char *str = (char *)arena_alloc(&arena, 2 * sizeof(char));
    str[0] = (char)i;
    str[1] = '\0';
    insert_entry(&arena, vocab, str, 1, (void *)(uintptr_t)i);
    insert_entry(&arena, r_vocab, &i, sizeof(uint32), str);
  }

  size_t vocab_len = 256;
  
  Chunks c = pretok_split(contents, f_size);
  uint32 tokens_len = 0;

  for (size_t i = 0; i < c.count; i++){
    tokens_len += c.lens[i];
  }
  uint32 *tokens = (uint32 *)arena_alloc(&arena, tokens_len * sizeof(uint32));
  size_t offset = 0;
  for (size_t i = 0; i < c.count; i++){
    encode_string(c.items[i], c.lens[i], tokens + offset);
    offset += c.lens[i];
  }

  while (vocab_len < MAX_VOCAB_LEN){

    uint32 **pairs_list = (uint32 **)arena_alloc(&arena, tokens_len * sizeof(uint32 *));
    HashTable *pairs = create_hash_table(&arena, 1000);
    size_t pair_bytes = 2 * sizeof(uint32);

    for (size_t i = 0; i < tokens_len - 1; i++){
      uint32 *pair = (uint32 *)arena_alloc(&arena, pair_bytes);
      pair[0] = tokens[i];
      pair[1] = tokens[i + 1];

      void *val_ptr = NULL;
      if (search_entry(pairs, pair, pair_bytes, &val_ptr) == 0){
        insert_entry(&arena, pairs, pair, pair_bytes, (void *)(uintptr_t)1);  
        pairs_list[i] = pair;
      } else {
        uintptr_t new_count = (uintptr_t)val_ptr + 1;
        insert_entry(&arena, pairs, pair, pair_bytes, (void *)new_count); 
      }
    }

    printf("starting max counting...\n");
    uint32 max_freq_pair[2] = {0, 0};
    uint32 max_freq = 1;

    for (uint32 i = 0; i < pairs->size; i++) {
      Entry *entry = pairs->buckets[i];
      while (entry) {
        if ((uint32)(uintptr_t)entry->value > max_freq){
          uint32 *p = (uint32 *)entry->key; 
          max_freq = (uint32)(uintptr_t)entry->value;
          max_freq_pair[0] = p[0];
          max_freq_pair[1] = p[1];
        }
        entry = entry->chain;
      }
    }

    printf("%d %d\n", max_freq_pair[0], max_freq_pair[1]);
    printf("%d\n", max_freq);

    if (max_freq < 2) break;

    vocab_len++;

    insert_entry(&arena, merges, max_freq_pair, 2 * sizeof(uint32), (void *)(uintptr_t)vocab_len);
    
    char *str1 = NULL;
    char *str2 = NULL;
    search_entry(r_vocab, &max_freq_pair[0], sizeof(uint32), (void **)&str1);
    search_entry(r_vocab, &max_freq_pair[1], sizeof(uint32), (void **)&str2);

    if (str1 && str2) {
      size_t len1 = strlen(str1);
      size_t len2 = strlen(str2);
      char *new_str = (char *)arena_alloc(&arena, len1 + len2 + 1);
      sprintf(new_str, "%s%s", str1, str2);

      insert_entry(&arena, vocab, new_str, len1 + len2, (void *)(uintptr_t)vocab_len);
      insert_entry(&arena, r_vocab, &vocab_len, sizeof(uint32), new_str);

      printf("Merged token #%zu: \"%s\" (from ID %u + ID %u)\n", vocab_len, new_str, max_freq_pair[0], max_freq_pair[1]);
    }
    printf("\n");
    
    size_t write_idx = 0;
    size_t read_idx = 0;

    while (read_idx < tokens_len) {
      if (read_idx < tokens_len - 1 && tokens[read_idx] == max_freq_pair[0] && tokens[read_idx + 1] == max_freq_pair[1]) {
        tokens[write_idx++] = vocab_len;
        read_idx += 2;
      } else {
        tokens[write_idx++] = tokens[read_idx++];
      }
    }
    tokens_len = write_idx;
  }
  chunks_free(&c);

  return 0;
}


