#include <stdio.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
	Matrix m;

	printf("matrix test\n");
	m = unitMatrix(3, 3);
	printMatrix(m);
	freeMatrix(m);

	return 0;
}
