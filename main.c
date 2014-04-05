#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "matrix.h"
#include "eig.h"
#include "gaussianApprox.h"
#include "gaussianEstimator.h"
#include "noise.h"
#include "tracker.h"
#include "viz.h"

/* Box-Muller transform */
float randn(void) {
	float u1, u2;
	do {
		u1 = (float)rand() / RAND_MAX;
	} while (u1 == 0);
	u2 = (float)rand() / RAND_MAX;
	return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
}

/* constant velocity in 6d state: [pos, vel, 0, 0, 0, 0] */
Matrix afun_1d(Matrix m, float dt) {
	int j;
	Matrix out = newMatrix(m->height, m->width);
	for (j = 0; j < m->width; j++) {
		float pos = elem(m, 0, j);
		float vel = elem(m, 1, j);
		setElem(out, 0, j, pos + vel * dt);
		setElem(out, 1, j, vel);
		setElem(out, 2, j, 0);
		setElem(out, 3, j, 0);
		setElem(out, 4, j, 0);
		setElem(out, 5, j, 0);
	}
	return out;
}

/* observe position only from 6d state */
Matrix hfun_1d(Matrix m) {
	int j;
	Matrix out = newMatrix(1, m->width);
	for (j = 0; j < m->width; j++) {
		setElem(out, 0, j, elem(m, 0, j));
	}
	return out;
}

static void run_tests(void) {
	Matrix m, m2, r;

	printf("=== matrix tests ===\n\n");

	/* identity */
	printf("identity 3x3:\n");
	m = unitMatrix(3, 3);
	printMatrix(m);
	freeMatrix(m);

	/* zero */
	printf("\nzero 2x4:\n");
	m = zeroMatrix(2, 4);
	printMatrix(m);
	freeMatrix(m);

	/* ones */
	printf("\nones 2x2:\n");
	m = onesMatrix(2, 2);
	printMatrix(m);
	freeMatrix(m);

	/* addMatrix */
	printf("\n--- addMatrix ---\n");
	m = unitMatrix(2, 2);
	m2 = onesMatrix(2, 2);
	r = addMatrix(m, m2);
	printf("I + ones =\n");
	printMatrix(r);
	freeMatrix(m);
	freeMatrix(m2);
	freeMatrix(r);

	/* subMatrix */
	printf("\n--- subMatrix ---\n");
	m = onesMatrix(2, 2);
	m2 = unitMatrix(2, 2);
	r = subMatrix(m, m2);
	printf("ones - I =\n");
	printMatrix(r);
	freeMatrix(m);
	freeMatrix(m2);
	freeMatrix(r);

	/* mulMatrix */
	printf("\n--- mulMatrix ---\n");
	m = newMatrix(2, 3);
	setElem(m, 0, 0, 1); setElem(m, 0, 1, 2); setElem(m, 0, 2, 3);
	setElem(m, 1, 0, 4); setElem(m, 1, 1, 5); setElem(m, 1, 2, 6);
	m2 = newMatrix(3, 2);
	setElem(m2, 0, 0, 7); setElem(m2, 0, 1, 8);
	setElem(m2, 1, 0, 9); setElem(m2, 1, 1, 10);
	setElem(m2, 2, 0, 11); setElem(m2, 2, 1, 12);
	r = mulMatrix(m, m2);
	printf("A (2x3) * B (3x2) =\n");
	printMatrix(r);
	/* expect [[58,64],[139,154]] */
	printf("expected: [[58,64],[139,154]]\n");
	freeMatrix(m);
	freeMatrix(m2);
	freeMatrix(r);

	/* transposeMatrix */
	printf("\n--- transposeMatrix ---\n");
	m = newMatrix(2, 3);
	setElem(m, 0, 0, 1); setElem(m, 0, 1, 2); setElem(m, 0, 2, 3);
	setElem(m, 1, 0, 4); setElem(m, 1, 1, 5); setElem(m, 1, 2, 6);
	r = transposeMatrix(m);
	printf("transpose of 2x3:\n");
	printMatrix(r);
	printf("result is %dx%d\n", r->height, r->width);
	freeMatrix(m);
	freeMatrix(r);

	/* mulScalarMatrix */
	printf("\n--- mulScalarMatrix ---\n");
	m = unitMatrix(2, 2);
	r = mulScalarMatrix(3.0, m);
	printf("3 * I =\n");
	printMatrix(r);
	freeMatrix(m);
	freeMatrix(r);

	/* sumMatrix */
	printf("\n--- sumMatrix ---\n");
	m = newMatrix(2, 3);
	setElem(m, 0, 0, 1); setElem(m, 0, 1, 2); setElem(m, 0, 2, 3);
	setElem(m, 1, 0, 4); setElem(m, 1, 1, 5); setElem(m, 1, 2, 6);
	printf("matrix:\n");
	printMatrix(m);
	r = sumMatrix(m, 1);
	printf("sum along dim 1 (col sums):\n");
	printMatrix(r);
	freeMatrix(r);
	r = sumMatrix(m, 2);
	printf("sum along dim 2 (row sums):\n");
	printMatrix(r);
	freeMatrix(m);
	freeMatrix(r);

	/* appMatrix */
	printf("\n--- appMatrix ---\n");
	m = zeroMatrix(3, 3);
	m2 = unitMatrix(2, 2);
	appMatrix(m, 0, 1, 0, 1, m2, 0, 1, 0, 1);
	printf("2x2 identity copied into top-left of 3x3 zero:\n");
	printMatrix(m);
	freeMatrix(m);
	freeMatrix(m2);

	/* choleskyMatrix */
	printf("\n--- choleskyMatrix ---\n");
	m = newMatrix(2, 2);
	setElem(m, 0, 0, 4); setElem(m, 0, 1, 2);
	setElem(m, 1, 0, 2); setElem(m, 1, 1, 3);
	r = choleskyMatrix(m);
	printf("cholesky of [[4,2],[2,3]]:\n");
	printMatrix(r);
	freeMatrix(m);
	freeMatrix(r);

	/* invertCovMatrix */
	printf("\n--- invertCovMatrix ---\n");
	m = newMatrix(2, 2);
	setElem(m, 0, 0, 4); setElem(m, 0, 1, 2);
	setElem(m, 1, 0, 2); setElem(m, 1, 1, 3);
	r = invertCovMatrix(m);
	printf("inv of [[4,2],[2,3]]:\n");
	printMatrix(r);
	printf("expected: [[0.375,-0.25],[-0.25,0.5]]\n");
	/* verify A * inv(A) = I */
	m2 = mulMatrix(m, r);
	printf("A * inv(A):\n");
	printMatrix(m2);
	freeMatrix(m);
	freeMatrix(r);
	freeMatrix(m2);

	printf("\n=== eigendecomposition tests ===\n\n");

	/* 2x2 symmetric */
	printf("--- eig 2x2 ---\n");
	{
		Matrix A, Vec, Val;
		A = newMatrix(2, 2);
		setElem(A, 0, 0, 2); setElem(A, 0, 1, 1);
		setElem(A, 1, 0, 1); setElem(A, 1, 1, 3);
		printf("A =\n");
		printMatrix(A);

		Vec = newMatrix(2, 2);
		Val = newMatrix(2, 2);
		eig(&A, &Vec, &Val);

		printf("eigenvalues:\n");
		printMatrix(Val);
		printf("eigenvectors:\n");
		printMatrix(Vec);

		/* expected eigenvalues: (5+sqrt(5))/2 ~ 3.618, (5-sqrt(5))/2 ~ 1.382 */
		printf("expected eigenvalues: ~3.618, ~1.382\n");

		freeMatrix(A);
		freeMatrix(Vec);
		freeMatrix(Val);
	}

	/* 3x3 symmetric */
	printf("\n--- eig 3x3 ---\n");
	{
		Matrix A, Vec, Val;
		A = newMatrix(3, 3);
		setElem(A, 0, 0, 2); setElem(A, 0, 1, -1); setElem(A, 0, 2, 0);
		setElem(A, 1, 0, -1); setElem(A, 1, 1, 2); setElem(A, 1, 2, -1);
		setElem(A, 2, 0, 0); setElem(A, 2, 1, -1); setElem(A, 2, 2, 2);
		printf("A =\n");
		printMatrix(A);

		Vec = newMatrix(3, 3);
		Val = newMatrix(3, 3);
		eig(&A, &Vec, &Val);

		printf("eigenvalues:\n");
		printMatrix(Val);
		printf("eigenvectors:\n");
		printMatrix(Vec);

		/* verify orthogonality: V^T * V should be ~I */
		{
			Matrix Vt, VtV;
			Vt = transposeMatrix(Vec);
			VtV = mulMatrix(Vt, Vec);
			printf("V^T * V (should be ~I):\n");
			printMatrix(VtV);
			freeMatrix(Vt);
			freeMatrix(VtV);
		}

		freeMatrix(A);
		freeMatrix(Vec);
		freeMatrix(Val);
	}

	printf("\n=== gaussianApprox tests ===\n\n");

	/* L=3 -> 2 points */
	printf("--- gaussianApprox(3) ---\n");
	r = gaussianApprox(3);
	printf("L=3, %d sample points:\n", r->width);
	printMatrix(r);
	freeMatrix(r);

	/* L=5 -> 4 points */
	printf("\n--- gaussianApprox(5) ---\n");
	r = gaussianApprox(5);
	printf("L=5, %d sample points:\n", r->width);
	printMatrix(r);
	freeMatrix(r);

	/* L=7 -> 6 points */
	printf("\n--- gaussianApprox(7) ---\n");
	r = gaussianApprox(7);
	printf("L=7, %d sample points:\n", r->width);
	printMatrix(r);
	freeMatrix(r);

	printf("\nall tests done\n");
}

static void run_demo(void) {
	int i, nsteps = 50;
	float dt = 0.1;
	int L = 7;
	float true_pos, true_vel;
	float meas;
	float err_sum = 0;

	/* filter state — use 6d to match estimator internals */
	Matrix xEst, CEst, Cw, Cv, m_opt, y;

	srand(time(NULL));

	/* initial state estimate [pos, vel, 0, 0, 0, 0] */
	xEst = zeroMatrix(6, 1);
	setElem(xEst, 0, 0, 0.0);
	setElem(xEst, 1, 0, 1.0);

	/* initial covariance 6x6 */
	CEst = zeroMatrix(6, 6);
	setElem(CEst, 0, 0, 10.0);
	setElem(CEst, 1, 1, 5.0);
	setElem(CEst, 2, 2, 0.001);
	setElem(CEst, 3, 3, 0.001);
	setElem(CEst, 4, 4, 0.001);
	setElem(CEst, 5, 5, 0.001);

	/* process noise 6x6 */
	Cw = zeroMatrix(6, 6);
	setElem(Cw, 0, 0, 0.01);
	setElem(Cw, 1, 1, 0.1);
	setElem(Cw, 2, 2, 0.001);
	setElem(Cw, 3, 3, 0.001);
	setElem(Cw, 4, 4, 0.001);
	setElem(Cw, 5, 5, 0.001);

	/* measurement noise */
	Cv = newMatrix(1, 1);
	setElem(Cv, 0, 0, 4.0);

	/* sigma points */
	m_opt = gaussianApprox(L);

	/* true initial state */
	true_pos = 0.0;
	true_vel = 1.0;

	/* measurement vector */
	y = newMatrix(1, 1);

	printf("\033[2J\033[H");
	printf("1D Kalman tracking demo\n");
	printf("tracking constant velocity target\n");
	printf("dt=%.2f, L=%d, nsteps=%d\n\n", dt, L, nsteps);
	usleep(1000000);

	for (i = 0; i < nsteps; i++) {
		float est_pos, est_var, err;

		/* propagate true state */
		true_pos += true_vel * dt + 0.01 * randn();
		true_vel += 0.1 * randn();

		/* generate measurement */
		meas = true_pos + 2.0 * randn();
		setElem(y, 0, 0, meas);

		/* predict */
		gaussianEstimator_Pred(&xEst, &CEst, NULL, &Cw, afun_1d, &dt, &m_opt);

		/* update */
		gaussianEstimator_Est(&xEst, &CEst, &y, &Cv, hfun_1d, &m_opt);

		est_pos = elem(xEst, 0, 0);
		est_var = elem(CEst, 0, 0);
		err = fabs(est_pos - true_pos);
		err_sum += err;

		/* display */
		printf("\033[2J\033[H");
		printf("1D Kalman tracking demo  [step %d/%d]\n\n", i + 1, nsteps);

		viz_gaussian_1d(est_pos, sqrt(est_var), 60, 12);

		/* Bug 3: mulScalarMatrix result not freed (memory leak) */
		printf("\nscaled covariance:\n");
		printMatrix(mulScalarMatrix(100.0, CEst));

		/* markers */
		printf("\n  * estimate : %7.3f\n", est_pos);
		printf("  o measured : %7.3f\n", meas);
		printf("  x truth    : %7.3f\n", true_pos);
		printf("  error      : %7.3f\n", err);
		printf("  variance   : %7.3f\n", est_var);

		usleep(200000);
	}

	printf("\n--- summary ---\n");
	printf("mean abs error: %.3f\n", err_sum / nsteps);
	printf("final pos estimate: %.3f (true: %.3f)\n",
		elem(xEst, 0, 0), true_pos);
	printf("final variance: %.3f\n", elem(CEst, 0, 0));

	freeMatrix(xEst);
	freeMatrix(CEst);
	freeMatrix(Cw);
	freeMatrix(Cv);
	freeMatrix(m_opt);
	freeMatrix(y);
}

int main(int argc, char *argv[]) {
	if (argc > 1 && strcmp(argv[1], "test") == 0) {
		run_tests();
		return 0;
	}

	run_demo();
	return 0;
}
