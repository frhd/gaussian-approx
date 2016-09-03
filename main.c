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
#include "sim.h"

#define MODE_2D   0
#define MODE_1D   1
#define MODE_TEST 2
#define MODE_GRID 3

typedef struct {
	int mode;
	int nsteps;
	float dt;
	int L;
	int seed;
	int quiet;
	int color;
	int trajectory;
	int interactive;
	int speed;
} Config;

static void print_usage(const char *prog) {
	printf("usage: %s [options]\n", prog);
	printf("  -m <mode>   demo mode: 2d, 1d, test, grid  (default: 2d)\n");
	printf("  -t <traj>   trajectory: circle, line, fig8, random (default: circle)\n");
	printf("  -n <steps>  number of steps                 (default: 100)\n");
	printf("  -d <dt>     time step                       (default: 0.1)\n");
	printf("  -L <level>  approximation level (3,5,7)     (default: 7)\n");
	printf("  -s <seed>   random seed                     (default: time-based)\n");
	printf("  -q          quiet mode (final result only)\n");
	printf("  -i          interactive mode (step with keyboard)\n");
	printf("  --speed <ms> animation delay in ms           (default: 100)\n");
	printf("  --no-color  disable ANSI colors\n");
	printf("  -h          show this help\n");
}

static int parse_trajectory(const char *s) {
	if (strcmp(s, "circle") == 0) return SIM_CIRCLE;
	if (strcmp(s, "line") == 0) return SIM_LINE;
	if (strcmp(s, "fig8") == 0) return SIM_FIGURE8;
	if (strcmp(s, "random") == 0) return SIM_RANDOM;
	return -1;
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

/* 2d constant velocity: state = [x, y, vx, vy, 0, 0] */
Matrix afun_2d(Matrix m, float dt) {
	int j;
	Matrix out = newMatrix(m->height, m->width);
	for (j = 0; j < m->width; j++) {
		float x = elem(m, 0, j);
		float y = elem(m, 1, j);
		float vx = elem(m, 2, j);
		float vy = elem(m, 3, j);
		setElem(out, 0, j, x + vx * dt);
		setElem(out, 1, j, y + vy * dt);
		setElem(out, 2, j, vx);
		setElem(out, 3, j, vy);
		setElem(out, 4, j, 0);
		setElem(out, 5, j, 0);
	}
	return out;
}

/* observe [x, y] from 6d state */
Matrix hfun_2d(Matrix m) {
	int j;
	Matrix out = newMatrix(2, m->width);
	for (j = 0; j < m->width; j++) {
		setElem(out, 0, j, elem(m, 0, j));
		setElem(out, 1, j, elem(m, 1, j));
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

/* process keyboard input, return 1 if should advance step, -1 if quit */
static int handle_input(int *paused, int *speed) {
	while (term_kbhit()) {
		int ch = term_getchar();
		if (ch == 'q' || ch == 'Q') return -1;
		if (ch == ' ' || ch == '\n' || ch == '\r') {
			*paused = 1;
			return 1;
		}
		if (ch == 'r' || ch == 'R') {
			*paused = 0;
		}
		if (ch == 'p' || ch == 'P') {
			*paused = 1;
		}
		if (ch == '+' || ch == '=') {
			*speed -= 20;
			if (*speed < 10) *speed = 10;
		}
		if (ch == '-') {
			*speed += 20;
			if (*speed > 500) *speed = 500;
		}
	}
	return 0;
}

static void run_demo(Config *cfg) {
	int i;
	float dt = cfg->dt;
	int L = cfg->L;
	int nsteps = cfg->nsteps;
	float true_pos, true_vel;
	float meas;
	float err_sum = 0;
	int paused, speed;
	float innov = 0;

	/* filter state — use 6d to match estimator internals */
	Matrix xEst, CEst, Cw, Cv, m_opt, y;

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

	paused = cfg->interactive;
	speed = cfg->speed;

	if (!cfg->quiet) {
		if (cfg->interactive)
			term_raw_mode();

		printf("\033[2J\033[H");
		printf("1D Kalman tracking demo\n");
		printf("tracking constant velocity target\n");
		printf("dt=%.2f, L=%d, nsteps=%d\n\n", dt, L, nsteps);
		usleep(1000000);
	}

	for (i = 0; i < nsteps; ) {
		float est_pos, est_var, err;
		int ret;

		if (!cfg->quiet && cfg->interactive) {
			/* wait for input in interactive mode */
			while (paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) goto done_1d;
				if (ret == 1) break;
				usleep(20000);
			}
			/* also check for input in run mode */
			if (!paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) goto done_1d;
			}
		}

		/* propagate true state */
		true_pos += true_vel * dt + 0.01 * randn();
		true_vel += 0.1 * randn();

		/* generate measurement */
		meas = true_pos + 2.0 * randn();
		setElem(y, 0, 0, meas);

		/* predicted measurement for innovation */
		{
			Matrix hpred = hfun_1d(xEst);
			innov = meas - elem(hpred, 0, 0);
			freeMatrix(hpred);
		}

		/* predict */
		gaussianEstimator_Pred(&xEst, &CEst, NULL, &Cw, afun_1d, &dt, &m_opt);

		/* update */
		gaussianEstimator_Est(&xEst, &CEst, &y, &Cv, hfun_1d, &m_opt);

		est_pos = elem(xEst, 0, 0);
		est_var = elem(CEst, 0, 0);
		err = fabs(est_pos - true_pos);
		err_sum += err;

		if (!cfg->quiet) {
			/* display */
			printf("\033[2J\033[H");
			printf("1D Kalman tracking demo  ");
			viz_color(COL_BOLD);
			printf("[step %d/%d]", i + 1, nsteps);
			viz_color(COL_RESET);
			printf("  ");
			if (paused) {
				viz_color(COL_YELLOW);
				printf("[PAUSED]");
			} else {
				viz_color(COL_GREEN);
				printf("[RUNNING]");
			}
			viz_color(COL_RESET);
			printf("\n\n");

			viz_gaussian_1d(est_pos, sqrt(est_var), 60, 12);

			/* Bug 3: mulScalarMatrix result not freed (memory leak) */
			printf("\nscaled covariance:\n");
			printMatrix(mulScalarMatrix(100.0, CEst));

			/* markers */
			printf("\n  ");
			viz_color(COL_CYAN);
			printf("*");
			viz_color(COL_RESET);
			printf(" estimate : %7.3f\n", est_pos);
			printf("  ");
			viz_color(COL_YELLOW);
			printf("o");
			viz_color(COL_RESET);
			printf(" measured : %7.3f\n", meas);
			printf("  ");
			viz_color(COL_GREEN);
			printf("x");
			viz_color(COL_RESET);
			printf(" truth    : %7.3f\n", true_pos);
			printf("  error      : %7.3f\n", err);
			printf("  variance   : %7.3f\n", est_var);
			printf("  velocity   : %7.3f\n", elem(xEst, 1, 0));
			printf("  innovation : %7.3f\n", innov);

			if (i == 0 && cfg->interactive) {
				viz_color(COL_DIM);
				printf("\n  [space] step  [r]un  [p]ause  [+/-] speed  [q]uit\n");
				viz_color(COL_RESET);
			}

			usleep(200000);
		}

		i++;
	}

done_1d:
	if (!cfg->quiet && cfg->interactive)
		term_restore();

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

static void run_demo_2d(Config *cfg) {
	int i;
	int nsteps = cfg->nsteps;
	float dt = cfg->dt;
	int L = cfg->L;
	float err_sum = 0;
	float xmin, xmax, ymin, ymax, margin;
	float trace_p0;
	int paused, speed;
	float innov_x = 0, innov_y = 0;

	Matrix xEst, CEst, Cw, Cv, m_opt, y;
	Matrix true_pos, meas;
	Scenario *scen;
	Grid g;

	/* generate scenario */
	scen = sim_create_scenario(cfg->trajectory, nsteps, dt, 2.0);
	true_pos = scen->true_pos;
	meas = scen->measurements;

	/* auto-scale grid bounds */
	xmin = xmax = elem(true_pos, 0, 0);
	ymin = ymax = elem(true_pos, 0, 1);
	for (i = 1; i < nsteps; i++) {
		float tx = elem(true_pos, i, 0);
		float ty = elem(true_pos, i, 1);
		if (tx < xmin) xmin = tx;
		if (tx > xmax) xmax = tx;
		if (ty < ymin) ymin = ty;
		if (ty > ymax) ymax = ty;
	}
	{
		float span = xmax - xmin;
		float yspan = ymax - ymin;
		if (yspan > span) span = yspan;
		margin = span * 0.25;
	}
	if (margin < 1.0) margin = 1.0;
	xmin -= margin; xmax += margin;
	ymin -= margin; ymax += margin;

	/* initial state estimate [x, y, vx, vy, 0, 0] */
	xEst = zeroMatrix(6, 1);
	setElem(xEst, 0, 0, elem(true_pos, 0, 0));
	setElem(xEst, 1, 0, elem(true_pos, 0, 1));
	setElem(xEst, 2, 0, 0.0);
	setElem(xEst, 3, 0, 0.5);

	/* initial covariance */
	CEst = zeroMatrix(6, 6);
	setElem(CEst, 0, 0, 10.0);
	setElem(CEst, 1, 1, 10.0);
	setElem(CEst, 2, 2, 5.0);
	setElem(CEst, 3, 3, 5.0);
	setElem(CEst, 4, 4, 0.001);
	setElem(CEst, 5, 5, 0.001);

	/* process noise */
	Cw = zeroMatrix(6, 6);
	setElem(Cw, 0, 0, 0.01);
	setElem(Cw, 1, 1, 0.01);
	setElem(Cw, 2, 2, 0.1);
	setElem(Cw, 3, 3, 0.1);
	setElem(Cw, 4, 4, 0.001);
	setElem(Cw, 5, 5, 0.001);

	/* measurement noise */
	Cv = zeroMatrix(2, 2);
	setElem(Cv, 0, 0, 4.0);
	setElem(Cv, 1, 1, 4.0);

	m_opt = gaussianApprox(L);
	y = newMatrix(2, 1);

	/* initial trace for convergence indicator */
	trace_p0 = elem(CEst, 0, 0) + elem(CEst, 1, 1);

	paused = cfg->interactive;
	speed = cfg->speed;

	if (!cfg->quiet) {
		if (cfg->interactive)
			term_raw_mode();

		printf("\033[2J\033[H");
		printf("2D Kalman tracking demo\n");
		printf("trajectory: %s\n", sim_trajectory_name(cfg->trajectory));
		printf("dt=%.2f, L=%d, nsteps=%d\n\n", dt, L, nsteps);

		/* show trajectory preview */
		viz_grid_init(&g, xmin, xmax, ymin, ymax);
		for (i = 0; i < nsteps; i++)
			viz_grid_point(&g, elem(true_pos, i, 0), elem(true_pos, i, 1), '.');
		viz_grid_print(&g);
		printf("\n");
		viz_color(COL_DIM);
		printf("  trajectory preview — press enter or wait...\n");
		viz_color(COL_RESET);
		usleep(2000000);
	}

	for (i = 0; i < nsteps; ) {
		float est_x, est_y, true_x, true_y, err;
		float trace_p;
		int ret;

		if (!cfg->quiet && cfg->interactive) {
			while (paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) goto done_2d;
				if (ret == 1) break;
				usleep(20000);
			}
			if (!paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) goto done_2d;
			}
		}

		true_x = elem(true_pos, i, 0);
		true_y = elem(true_pos, i, 1);

		/* measurement */
		setElem(y, 0, 0, elem(meas, i, 0));
		setElem(y, 1, 0, elem(meas, i, 1));

		/* compute innovation before prediction */
		{
			Matrix hpred = hfun_2d(xEst);
			innov_x = elem(meas, i, 0) - elem(hpred, 0, 0);
			innov_y = elem(meas, i, 1) - elem(hpred, 1, 0);
			freeMatrix(hpred);
		}

		/* predict */
		gaussianEstimator_Pred(&xEst, &CEst, NULL, &Cw, afun_2d, &dt, &m_opt);

		/* update */
		gaussianEstimator_Est(&xEst, &CEst, &y, &Cv, hfun_2d, &m_opt);

		est_x = elem(xEst, 0, 0);
		est_y = elem(xEst, 1, 0);
		err = sqrt((est_x - true_x) * (est_x - true_x) +
		           (est_y - true_y) * (est_y - true_y));
		err_sum += err;

		trace_p = elem(CEst, 0, 0) + elem(CEst, 1, 1);

		if (!cfg->quiet) {
			/* draw grid */
			viz_grid_init(&g, xmin, xmax, ymin, ymax);

			/* draw covariance ellipse first (so markers draw on top) */
			viz_grid_ellipse(&g, est_x, est_y, CEst, '~');

			/* plot true trajectory up to this step */
			{
				int k;
				for (k = 0; k <= i; k++)
					viz_grid_point(&g, elem(true_pos, k, 0), elem(true_pos, k, 1), '.');
			}

			/* plot measurements up to this step */
			{
				int k;
				for (k = 0; k <= i; k++)
					viz_grid_point(&g, elem(meas, k, 0), elem(meas, k, 1), 'o');
			}

			/* plot estimate */
			viz_grid_point(&g, est_x, est_y, '+');

			/* display */
			printf("\033[2J\033[H");
			printf("2D Kalman tracking  ");
			viz_color(COL_BOLD);
			printf("[step %d/%d]", i + 1, nsteps);
			viz_color(COL_RESET);
			printf("  ");
			if (paused) {
				viz_color(COL_YELLOW);
				printf("[PAUSED]");
			} else {
				viz_color(COL_GREEN);
				printf("[RUNNING]");
			}
			viz_color(COL_RESET);
			printf("\n\n");
			viz_grid_print(&g);

			printf("\n  ");
			viz_color(COL_GREEN);
			printf(".");
			viz_color(COL_RESET);
			printf(" truth    (%7.2f, %7.2f)\n", true_x, true_y);
			printf("  ");
			viz_color(COL_YELLOW);
			printf("o");
			viz_color(COL_RESET);
			printf(" measured (%7.2f, %7.2f)\n", elem(meas, i, 0), elem(meas, i, 1));
			printf("  ");
			viz_color(COL_CYAN);
			printf("+");
			viz_color(COL_RESET);
			printf(" estimate (%7.2f, %7.2f)\n", est_x, est_y);
			printf("  ");
			viz_color(COL_DIM);
			printf("~");
			viz_color(COL_RESET);
			printf(" covariance ellipse\n");
			{
				float rmse = err_sum / (i + 1);
				printf("  error: %.3f   rmse: ", err);
				if (rmse < 2.0) viz_color(COL_GREEN);
				else if (rmse < 5.0) viz_color(COL_YELLOW);
				else viz_color(COL_RED);
				printf("%.3f", rmse);
				viz_color(COL_RESET);
				printf("\n");
			}

			/* state vector */
			printf("  vel: (%.3f, %.3f)  ", elem(xEst, 2, 0), elem(xEst, 3, 0));
			printf("innov: (%.3f, %.3f)\n", innov_x, innov_y);

			/* covariance and convergence */
			printf("  cov diag: %.3f, %.3f  trace(P): %.3f\n",
				elem(CEst, 0, 0), elem(CEst, 1, 1), trace_p);
			printf("  convergence: ");
			viz_convergence_bar(trace_p, trace_p0, 20);
			printf("\n");

			if (i == 0 && cfg->interactive) {
				viz_color(COL_DIM);
				printf("\n  [space] step  [r]un  [p]ause  [+/-] speed  [q]uit\n");
				viz_color(COL_RESET);
			}

			usleep(100000);
		}

		i++;
	}

done_2d:
	if (!cfg->quiet && cfg->interactive)
		term_restore();

	printf("\n--- summary ---\n");
	printf("mean rmse: %.3f\n", err_sum / nsteps);
	printf("final estimate: (%.3f, %.3f)\n", elem(xEst, 0, 0), elem(xEst, 1, 0));
	printf("final truth:    (%.3f, %.3f)\n",
		elem(true_pos, nsteps - 1, 0), elem(true_pos, nsteps - 1, 1));

	freeMatrix(xEst);
	freeMatrix(CEst);
	freeMatrix(Cw);
	freeMatrix(Cv);
	freeMatrix(m_opt);
	freeMatrix(y);
	sim_free_scenario(scen);
}

static void run_grid_demo(void) {
	int k, npts = 50;
	Grid g;
	Matrix xs, ys;

	xs = newMatrix(1, npts);
	ys = newMatrix(1, npts);
	for (k = 0; k < npts; k++) {
		float t = -3.14159 + k * 2.0 * 3.14159 / (npts - 1);
		setElem(xs, 0, k, t);
		setElem(ys, 0, k, sin(t));
	}

	viz_grid_init(&g, -3.14159, 3.14159, -1.2, 1.2);
	viz_grid_trajectory(&g, xs, ys, '*');

	/* add a circle */
	for (k = 0; k < 60; k++) {
		float angle = k * 2.0 * 3.14159 / 60;
		viz_grid_point(&g, cos(angle), sin(angle), 'o');
	}

	/* some individual test points */
	viz_grid_point(&g, 0.0, 0.0, '+');
	viz_grid_point(&g, -3.0, 0.0, 'L');
	viz_grid_point(&g, 3.0, 0.0, 'R');
	viz_grid_point(&g, 0.0, 1.0, 'T');
	viz_grid_point(&g, 0.0, -1.0, 'B');

	printf("2d grid renderer test\n\n");
	viz_grid_print(&g);
	printf("\nlegend: * sine wave  o unit circle  + origin  L/R/T/B edges\n");

	/* free trajectory matrices */
	freeMatrix(xs);
	freeMatrix(ys);
}

static int parse_mode(const char *s) {
	if (strcmp(s, "2d") == 0) return MODE_2D;
	if (strcmp(s, "1d") == 0) return MODE_1D;
	if (strcmp(s, "test") == 0) return MODE_TEST;
	if (strcmp(s, "grid") == 0) return MODE_GRID;
	return -1;
}

int main(int argc, char *argv[]) {
	int opt, i;
	Config cfg;

	/* defaults */
	cfg.mode = MODE_2D;
	cfg.nsteps = 100;
	cfg.dt = 0.1;
	cfg.L = 7;
	cfg.seed = -1;
	cfg.quiet = 0;
	cfg.color = 1;
	cfg.trajectory = SIM_CIRCLE;
	cfg.interactive = 0;
	cfg.speed = 100;

	/* check for gnu-style long options before getopt */
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--no-color") == 0) {
			cfg.color = 0;
			memmove(&argv[i], &argv[i + 1], (argc - i - 1) * sizeof(char *));
			argc--;
			i--;
		} else if (strcmp(argv[i], "--speed") == 0 && i + 1 < argc) {
			cfg.speed = atoi(argv[i + 1]);
			if (cfg.speed <= 0) cfg.speed = 100;
			memmove(&argv[i], &argv[i + 2], (argc - i - 2) * sizeof(char *));
			argc -= 2;
			i--;
		}
	}

	while ((opt = getopt(argc, argv, "m:t:n:d:L:s:qih")) != -1) {
		switch (opt) {
		case 'm':
			cfg.mode = parse_mode(optarg);
			if (cfg.mode < 0) {
				fprintf(stderr, "unknown mode: %s\n", optarg);
				print_usage(argv[0]);
				return 1;
			}
			break;
		case 't':
			cfg.trajectory = parse_trajectory(optarg);
			if (cfg.trajectory < 0) {
				fprintf(stderr, "unknown trajectory: %s\n", optarg);
				print_usage(argv[0]);
				return 1;
			}
			break;
		case 'n':
			cfg.nsteps = atoi(optarg);
			if (cfg.nsteps <= 0) {
				fprintf(stderr, "steps must be > 0\n");
				return 1;
			}
			break;
		case 'd':
			cfg.dt = atof(optarg);
			if (cfg.dt <= 0) {
				fprintf(stderr, "dt must be > 0\n");
				return 1;
			}
			break;
		case 'L':
			cfg.L = atoi(optarg);
			if (cfg.L != 3 && cfg.L != 5 && cfg.L != 7) {
				fprintf(stderr, "L must be 3, 5, or 7\n");
				return 1;
			}
			break;
		case 's':
			cfg.seed = atoi(optarg);
			break;
		case 'q':
			cfg.quiet = 1;
			break;
		case 'i':
			cfg.interactive = 1;
			break;
		case 'h':
			print_usage(argv[0]);
			return 0;
		default:
			print_usage(argv[0]);
			return 1;
		}
	}

	/* seed rng */
	if (cfg.seed >= 0)
		srand(cfg.seed);
	else
		srand(time(NULL));

	/* disable colors if not a tty or --no-color given */
	if (!cfg.color || !isatty(STDOUT_FILENO))
		viz_color_enabled = 0;

	/* print config summary */
	if (!cfg.quiet) {
		const char *mnames[] = {"2d", "1d", "test", "grid"};
		printf("vizga: mode=%s traj=%s steps=%d dt=%.2f L=%d seed=%s color=%s%s speed=%dms\n",
			mnames[cfg.mode], sim_trajectory_name(cfg.trajectory),
			cfg.nsteps, cfg.dt, cfg.L,
			cfg.seed >= 0 ? "fixed" : "time",
			viz_color_enabled ? "on" : "off",
			cfg.interactive ? " interactive" : "",
			cfg.speed);
	}

	switch (cfg.mode) {
	case MODE_TEST:
		run_tests();
		break;
	case MODE_GRID:
		run_grid_demo();
		break;
	case MODE_1D:
		run_demo(&cfg);
		break;
	case MODE_2D:
	default:
		run_demo_2d(&cfg);
		break;
	}

	return 0;
}
