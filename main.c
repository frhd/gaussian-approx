#include <stdio.h>
#include "matrix.h"

int main(int argc, char *argv[]) {
	Matrix m, m2, r;

	printf("=== matrix tests ===\n\n");

	printf("identity 3x3:\n");
	m = unitMatrix(3, 3);
	printMatrix(m);
	freeMatrix(m);

	printf("\nzero 2x4:\n");
	m = zeroMatrix(2, 4);
	printMatrix(m);
	freeMatrix(m);

	printf("\nones 2x2:\n");
	m = onesMatrix(2, 2);
	printMatrix(m);
	freeMatrix(m);

	printf("\n--- addMatrix ---\n");
	m = unitMatrix(2, 2);
	m2 = onesMatrix(2, 2);
	r = addMatrix(m, m2);
	printf("I + ones =\n");
	printMatrix(r);
	freeMatrix(m);
	freeMatrix(m2);
	freeMatrix(r);

	return 0;
}
