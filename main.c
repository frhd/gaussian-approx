#include <stdio.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
	Matrix m;

	printf("=== matrix tests ===\n\n");

	printf("identity 3x3:\n");
	m = unitMatrix(3, 3);
	printMatrix(m);
	freeMatrix(m);

	printf("\nzero 2x4:\n");
	m = zeroMatrix(2, 4);
	printMatrix(m);
	freeMatrix(m);

	return 0;
}
