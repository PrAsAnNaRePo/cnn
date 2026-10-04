#include "pretok.h"
#include "types.h"
#include "utils.c"
#include "arena.h"
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
  }

  size_t vocab_len = 256;
  
  Chunks c = pretok_split(contents, f_size);
  uint32 tokens_len = 0;

  for (size_t i = 0; i < c.count; i++){
    tokens_len += c.lens[i];
  }
  printf("initial tokens_len: %u\n", tokens_len);

  uint32 *tokens = (uint32 *)arena_alloc(&arena, tokens_len * sizeof(uint32));
  for (size_t i = 0; i < c.count; i++){
    encode_string(c.items[i], c.lens[i], tokens + (i * c.lens[i]));
  }

  while (vocab_len > MAX_VOCAB_LEN){
    
  }

  chunks_free(&c);

  return 0;
}


