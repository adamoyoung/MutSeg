#include <stdio.h>
#include <stdlib.h>

void seed(int seed) {
	srand(seed);
}

int randint(void) {
	return rand();
}