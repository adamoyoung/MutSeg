CC = gcc
CFLAGS += -std=gnu11 -Wall -Werror -O3 -g3 -fopenmp
LDFLAGS += -lm -fopenmp

all: segmentation

segmentation.o: k_seg.h
k_seg.o: k_seg.h
# k_seg_mp.o: k_seg.h

segmentation: segmentation.o k_seg.o
	$(CC) $^ -o $@ $(LDFLAGS)

# segmentation_mp: segmentation.o k_seg_mp.o
# 	$(CC) $^ -o $@ $(LDFLAGS)

k_seg: k_seg.o
	$(CC) $^ -o $@ $(LDFLAGS)

# k_seg_mp: k_seg_mp.o
# 	$(CC) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o segmentation k_seg

clean_data:
	rm -f E_f_file.dat S_s_file.dat