CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -I.
SRCS = main.c src/arena.c src/tensor.c src/autograd.c 
TARGET = main

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
