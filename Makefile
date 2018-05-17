CC = gcc
CFLAGS += -std=gnu11 -Wall -Werror -O3
LDFLAGS += -lm

all: segmentation

segmentation.o: k_seg.h
k_seg.o: k_seg.h

segmentation: segmentation.o k_seg.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o segmentation