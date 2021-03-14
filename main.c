#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "matrix.h"
#include "eig.h"
#include "gaussianApprox.h"
#include "gaussianEstimator.h"
#include "noise.h"
#include "tracker.h"
#include "viz.h"
#include "sim.h"
#include "export.h"

#define MODE_2D    0
#define MODE_1D    1
#define MODE_TEST  2
#define MODE_GRID  3
#define MODE_COMPARE 6
#define MODE_MULTI 4
#define MODE_ROT   5

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
	int loop;
	char *outfile;
	int ntargets;
	int verbose;
	int metrics;
} Config;

static void print_usage(const char *prog) {
	printf("usage: %s [options]\n", prog);
	printf("  -m <mode>   demo mode: 2d, 1d, multi, rot, test, grid, compare (default: 2d)\n");
	printf("  -t <traj>   trajectory: circle, line, fig8, random (default: circle)\n");
	printf("  -n <steps>  number of steps                 (default: 100)\n");
	printf("  -d <dt>     time step                       (default: 0.1)\n");
	printf("  -L <level>  approximation level (3,5,7)     (default: 7)\n");
	printf("  -s <seed>   random seed                     (default: time-based)\n");
	printf("  -k <num>    number of targets (1-4)         (default: 2)\n");
	printf("  -o <file>   export data to CSV file\n");
	printf("  -q          quiet mode (final result only)\n");
	printf("  -i          interactive mode (step with keyboard)\n");
	printf("  -v          verbose mode (show sigma points)\n");
	printf("  --speed <ms> animation delay in ms           (default: 100)\n");
	printf("  --no-color  disable ANSI colors\n");
	printf("  --loop      auto-restart with new seed when done\n");
	printf("  --metrics   show convergence metrics (NIS, sparkline)\n");
	printf("  -h          show this help\n");
}

static int parse_trajectory(const char *s) {
	if (strcmp(s, "circle") == 0) return SIM_CIRCLE;
	if (strcmp(s, "line") == 0) return SIM_LINE;
	if (strcmp(s, "fig8") == 0) return SIM_FIGURE8;
	if (strcmp(s, "random") == 0) return SIM_RANDOM;
	return -1;
}

/* constant velocity in 6d state: [pos, vel, 0, 0, 0, 0]
 * padded to 6d because gaussianEstimator assumes it internally */
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

/* 2d constant velocity model: state = [x, y, vx, vy, 0, 0]
 * F matrix is [[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]]
 * padded to 6d because gaussianEstimator_Est has hardcoded indices */
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

/* observe position [x, y] from 6d state — H = [[1,0,0,0,0,0], [0,1,0,0,0,0]] */
Matrix hfun_2d(Matrix m) {
	int j;
	Matrix out = newMatrix(2, m->width);
	for (j = 0; j < m->width; j++) {
		setElem(out, 0, j, elem(m, 0, j));
		setElem(out, 1, j, elem(m, 1, j));
	}
	return out;
}

/* observe 3d position from 12-state: extract rows 0-2 */
Matrix hfun_3d(Matrix m) {
	int j;
	Matrix out = newMatrix(3, m->width);
	for (j = 0; j < m->width; j++) {
		setElem(out, 0, j, elem(m, 0, j));
		setElem(out, 1, j, elem(m, 1, j));
		setElem(out, 2, j, elem(m, 2, j));
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

/* compute normalized innovation squared (NIS) for 2D filter
 * NIS = innovation^T * inv(Cy) * innovation
 * where Cy is the predicted measurement covariance */
static float compute_nis_2d(float innov_x, float innov_y, Matrix Cy_pred) {
	Matrix innov_vec, inv_cy, temp, nis_mat;
	float nis;

	/* innovation vector [innov_x; innov_y] */
	innov_vec = newMatrix(2, 1);
	setElem(innov_vec, 0, 0, innov_x);
	setElem(innov_vec, 1, 0, innov_y);

	/* inverse of predicted measurement covariance */
	inv_cy = invertCovMatrix(Cy_pred);

	/* nis = innov^T * inv(Cy) * innov */
	temp = mulMatrix(inv_cy, innov_vec);
	nis_mat = mulMatrix(transposeMatrix(innov_vec), temp);
	nis = elem(nis_mat, 0, 0);

	freeMatrix(innov_vec);
	freeMatrix(inv_cy);
	freeMatrix(temp);
	freeMatrix(nis_mat);

	return nis;
}

/* process keyboard input
 * returns: 1=advance step, 0=nothing, -1=quit, -2=restart, -3=next traj */
static int handle_input(int *paused, int *speed) {
	while (term_kbhit()) {
		int ch = term_getchar();
		if (ch == 'q' || ch == 'Q') return -1;
		if (ch == ' ' || ch == '\n' || ch == '\r') {
			*paused = 1;
			return 1;
		}
		if (ch == 'r' || ch == 'R') {
			return -2;
		}
		if (ch == 'p' || ch == 'P') {
			*paused = 1;
		}
		if (ch == 'n' || ch == 'N') {
			return -3;
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

/* 1d layout — curve at top, state info below */
static void render_frame_1d(float est_pos, float est_var, float meas,
	float true_pos, float err, float innov, int step, int nsteps,
	float vel, int paused, int interactive, float elapsed) {

	viz_clear_screen();

	/* header */
	viz_cursor_move(1, 1);
	printf("1D Kalman tracking demo  ");
	viz_color(COL_BOLD);
	printf("[step %d/%d]", step + 1, nsteps);
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
	viz_color(COL_DIM);
	printf("  %.1fs", elapsed);
	viz_color(COL_RESET);

	/* gaussian curve — printed sequentially (uses printf directly) */
	viz_cursor_move(3, 1);
	viz_gaussian_1d(est_pos, sqrt(est_var), 60, 12);

	/* state info below the curve */
	/* cursor is already below the curve output */
	printf("\n");

	printf("  ");
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
	printf("  velocity   : %7.3f\n", vel);
	printf("  innovation : %7.3f\n", innov);

	printf("  ");
	viz_progress_bar(step, nsteps, 30);
	printf("\n");

	if (step == 0 && interactive) {
		viz_color(COL_DIM);
		printf("\n  [space] step  [r]un  [p]ause  [+/-] speed  [q]uit\n");
		viz_color(COL_RESET);
	}

	fflush(stdout);
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
	struct timeval t_start, t_now;

	if (nsteps <= 0) {
		printf("nothing to do (nsteps=0)\n");
		return;
	}

	/* filter state — use 6d to match estimator internals */
	Matrix xEst, CEst, Cw, Cv, m_opt, y;
	FILE *expf = NULL;

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

	/* open export file if requested */
	if (cfg->outfile) {
		expf = export_open(cfg->outfile);
		if (expf) {
			export_header_1d(expf);
			printf("Exporting to: %s\n", cfg->outfile);
		}
	}

	paused = cfg->interactive;
	speed = cfg->speed;

	if (!cfg->quiet) {
		if (cfg->interactive)
			term_raw_mode();

		viz_clear_screen();
		printf("1D Kalman tracking demo\n");
		printf("tracking constant velocity target\n");
		printf("dt=%.2f, L=%d, nsteps=%d\n\n", dt, L, nsteps);
		usleep(1000000);
	}

	gettimeofday(&t_start, NULL);

	for (i = 0; i < nsteps; ) {
		float est_pos, est_var, err, elapsed;
		int ret;

		if (!cfg->quiet && cfg->interactive) {
			/* wait for input in interactive mode */
			while (paused) {
				ret = handle_input(&paused, &speed);
				if (ret < 0) goto done_1d;
				if (ret == 1) break;
				usleep(20000);
			}
			/* also check for input in run mode */
			if (!paused) {
				ret = handle_input(&paused, &speed);
				if (ret < 0) goto done_1d;
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

		if (expf) {
			export_row_1d(expf, i, i * dt,
				true_pos, meas, est_pos, elem(xEst, 1, 0),
				est_var, err_sum / (i + 1));
		}

		if (!cfg->quiet) {
			gettimeofday(&t_now, NULL);
			elapsed = (t_now.tv_sec - t_start.tv_sec) +
				(t_now.tv_usec - t_start.tv_usec) / 1e6;
			render_frame_1d(est_pos, est_var, meas, true_pos, err, innov,
				i, nsteps, elem(xEst, 1, 0), paused, cfg->interactive, elapsed);

			usleep(speed * 1000);
		}

		i++;
	}

done_1d:
	if (!cfg->quiet && cfg->interactive)
		term_restore();

	printf("\n--- summary ---\n");
	printf("mean abs error: %.3f\n", err_sum / (i > 0 ? i : 1));
	printf("final pos estimate: %.3f (true: %.3f)\n",
		elem(xEst, 0, 0), true_pos);
	printf("final variance: %.3f\n", elem(CEst, 0, 0));

	if (expf)
		export_close(expf);

	freeMatrix(xEst);
	freeMatrix(CEst);
	freeMatrix(Cw);
	freeMatrix(Cv);
	freeMatrix(m_opt);
	freeMatrix(y);
}

/* 2d panel layout — grid on left, state info on right */
static void render_frame_2d(Grid *g, Config *cfg, int step, int nsteps,
	float true_x, float true_y, float meas_x, float meas_y,
	float est_x, float est_y, float vx, float vy,
	float cov_xx, float cov_yy, float trace_p, float trace_p0,
	float rmse, float err, float innov_x, float innov_y,
	float nis, float nis_avg, int paused, float elapsed,
	float *innov_hist, float *rmse_hist, int hist_idx) {
	int wide = viz_term_width() >= 80;
	Panel sp;
	char buf[64];

	viz_clear_screen();

	/* header line */
	viz_cursor_move(1, 1);
	printf("2D Kalman tracking  ");
	viz_color(COL_BOLD);
	printf("[step %d/%d]", step + 1, nsteps);
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
	viz_color(COL_DIM);
	printf("  %.1fs", elapsed);
	viz_color(COL_RESET);

	/* grid at row 3 */
	viz_grid_print_at(g, 3, 1);

	if (wide) {
		/* state panel to the right of the grid */
		sp.row = 3;
		sp.col = 7 + GRID_W + 2;  /* after y-labels + grid + gap */
		sp.width = 18;
		sp.height = 19;
		viz_panel_border(&sp);

		snprintf(buf, sizeof(buf), "  %s", sim_trajectory_name(cfg->trajectory));
		viz_panel_text(&sp, 0, buf);

		snprintf(buf, sizeof(buf), " Step %d/%d", step + 1, nsteps);
		viz_panel_text(&sp, 2, buf);

		snprintf(buf, sizeof(buf), " x:  %7.2f", est_x);
		viz_panel_text(&sp, 4, buf);
		snprintf(buf, sizeof(buf), " y:  %7.2f", est_y);
		viz_panel_text(&sp, 5, buf);
		snprintf(buf, sizeof(buf), " vx: %7.2f", vx);
		viz_panel_text(&sp, 6, buf);
		snprintf(buf, sizeof(buf), " vy: %7.2f", vy);
		viz_panel_text(&sp, 7, buf);

		viz_cursor_move(sp.row + 1 + 9, sp.col + 1);
		{
			float sx = sqrt(cov_xx > 0 ? cov_xx : 0);
			float sy = sqrt(cov_yy > 0 ? cov_yy : 0);
			snprintf(buf, sizeof(buf), " \317\203x: %7.2f", sx);
			viz_panel_text(&sp, 9, buf);
			snprintf(buf, sizeof(buf), " \317\203y: %7.2f", sy);
			viz_panel_text(&sp, 10, buf);
		}

		viz_cursor_move(sp.row + 1 + 12, sp.col + 1);
		printf(" RMSE: ");
		if (rmse < 2.0) viz_color(COL_GREEN);
		else if (rmse < 5.0) viz_color(COL_YELLOW);
		else viz_color(COL_RED);
		printf("%5.2f", rmse);
		viz_color(COL_RESET);

		snprintf(buf, sizeof(buf), " err:  %5.2f", err);
		viz_panel_text(&sp, 13, buf);

		/* NIS metrics display */
		if (cfg->metrics) {
			viz_cursor_move(sp.row + 1 + 14, sp.col + 1);
			printf(" NIS: ");
			if (nis < 2.0) viz_color(COL_GREEN);
			else if (nis < 5.0) viz_color(COL_YELLOW);
			else viz_color(COL_RED);
			printf("%5.2f", nis);
			viz_color(COL_RESET);
			viz_cursor_move(sp.row + 1 + 15, sp.col + 1);
			viz_color(COL_DIM);
			printf(" avg: %5.2f", nis_avg);
			viz_color(COL_RESET);

			/* sparkline of innovation history */
			if (hist_idx > 0) {
				viz_cursor_move(sp.row + 1 + 16, sp.col + 1);
				viz_sparkline(innov_hist, hist_idx, sp.width - 2);
			}

			/* sparkline of RMSE history */
			if (hist_idx > 0) {
				viz_cursor_move(sp.row + 1 + 17, sp.col + 1);
				viz_sparkline(rmse_hist, hist_idx, sp.width - 2);
			}
		}
	}

	/* legend and extra info below grid */
	{
		int brow = 3 + GRID_H + 2;

		viz_cursor_move(brow, 1);
		printf("  ");
		viz_color(COL_GREEN);
		printf(".");
		viz_color(COL_RESET);
		printf(" truth    (%7.2f, %7.2f)", true_x, true_y);

		viz_cursor_move(brow + 1, 1);
		printf("  ");
		viz_color(COL_YELLOW);
		printf("o");
		viz_color(COL_RESET);
		printf(" measured (%7.2f, %7.2f)", meas_x, meas_y);

		viz_cursor_move(brow + 2, 1);
		printf("  ");
		viz_color(COL_CYAN);
		printf("+");
		viz_color(COL_RESET);
		printf(" estimate (%7.2f, %7.2f)", est_x, est_y);

		viz_cursor_move(brow + 3, 1);
		printf("  ");
		viz_color(COL_DIM);
		printf("~");
		viz_color(COL_RESET);
		printf(" covariance ellipse");

		if (!wide) {
			/* fallback: show state info below for narrow terminals */
			viz_cursor_move(brow + 4, 1);
			printf("  vel: (%.3f, %.3f)  innov: (%.3f, %.3f)",
				vx, vy, innov_x, innov_y);
			viz_cursor_move(brow + 5, 1);
			printf("  cov diag: %.3f, %.3f  trace(P): %.3f",
				cov_xx, cov_yy, trace_p);
			viz_cursor_move(brow + 6, 1);
			printf("  RMSE: ");
			if (rmse < 2.0) viz_color(COL_GREEN);
			else if (rmse < 5.0) viz_color(COL_YELLOW);
			else viz_color(COL_RED);
			printf("%.3f", rmse);
			viz_color(COL_RESET);

			viz_cursor_move(brow + 7, 1);
			printf("  convergence: ");
			viz_convergence_bar(trace_p, trace_p0, 20);

			/* NIS metrics for narrow terminals */
			if (cfg->metrics) {
				viz_cursor_move(brow + 8, 1);
				printf("  NIS: ");
				if (nis < 2.0) viz_color(COL_GREEN);
				else if (nis < 5.0) viz_color(COL_YELLOW);
				else viz_color(COL_RED);
				printf("%.3f", nis);
				viz_color(COL_RESET);
				viz_color(COL_DIM);
				printf("  avg: %.3f", nis_avg);
				viz_color(COL_RESET);
			}
		} else {
			viz_cursor_move(brow + 4, 1);
			printf("  innov: (%.3f, %.3f)  convergence: ", innov_x, innov_y);
			viz_convergence_bar(trace_p, trace_p0, 20);
		}
	}

	/* progress bar */
	{
		int prow = 3 + GRID_H + 2 + (wide ? 5 : 8);
		viz_cursor_move(prow, 3);
		viz_progress_bar(step, nsteps, 30);
	}

	/* controls help on first frame */
	if (step == 0 && cfg->interactive) {
		int hrow = 3 + GRID_H + (wide ? 8 : 11) + (cfg->metrics ? (wide ? 0 : 2) : 0);
		viz_cursor_move(hrow, 1);
		viz_color(COL_DIM);
		printf("  [space] step  [p]ause  [r]estart  [n]ext  [+/-] speed  [q]uit");
		viz_color(COL_RESET);
	}

	/* move cursor to bottom so output doesn't mess up layout */
	viz_cursor_move(3 + GRID_H + (wide ? 10 : 13) + (cfg->metrics ? (wide ? 0 : 2) : 0), 1);
	fflush(stdout);
}

static void run_demo_2d(Config *cfg) {
	int i;
	int nsteps = cfg->nsteps;
	float dt = cfg->dt;
	int L = cfg->L;
	float err_sum = 0, err_max = 0;
	float xmin, xmax, ymin, ymax, margin;
	int traj_type = cfg->trajectory;

	if (nsteps <= 0) {
		printf("nothing to do (nsteps=0)\n");
		return;
	}
	float trace_p0;
	int paused, speed;
	float innov_x = 0, innov_y = 0;
	struct timeval t_start, t_now;
	int action;  /* 0=normal end, -1=quit, -2=restart, -3=next */

	/* metrics tracking */
	float innov_hist[50] = {0};
	float rmse_hist[50] = {0};
	float nis, nis_sum = 0;
	int innov_idx = 0, rmse_idx = 0;

	Matrix xEst, CEst, Cw, Cv, m_opt, y;
	Matrix true_pos, meas;
	Scenario *scen;
	Grid g;
	FILE *expf = NULL;

	/* process noise — constant across restarts */
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

	/* open export file if requested */
	if (cfg->outfile) {
		expf = export_open(cfg->outfile);
		if (expf) {
			export_header_2d(expf);
			printf("Exporting to: %s\n", cfg->outfile);
		}
	}

	speed = cfg->speed;

	if (!cfg->quiet && cfg->interactive)
		term_raw_mode();

restart_2d:
	action = 0;
	err_sum = 0;
	err_max = 0;
	innov_x = 0;
	innov_y = 0;

	/* initialize metrics */
	nis_sum = 0;
	innov_idx = 0;
	rmse_idx = 0;
	for (i = 0; i < 50; i++) {
		innov_hist[i] = 0;
		rmse_hist[i] = 0;
	}

	/* generate scenario */
	scen = sim_create_scenario(traj_type, nsteps, dt, 2.0);
	if (!scen) {
		fprintf(stderr, "failed to create scenario\n");
		goto cleanup_2d;
	}
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

	/* initial trace for convergence indicator */
	trace_p0 = elem(CEst, 0, 0) + elem(CEst, 1, 1);

	paused = cfg->interactive;

	if (!cfg->quiet) {
		viz_clear_screen();
		printf("2D Kalman tracking demo\n");
		printf("trajectory: %s\n", sim_trajectory_name(traj_type));
		printf("dt=%.2f, L=%d, nsteps=%d\n\n", dt, L, nsteps);

		/* show trajectory preview */
		viz_grid_init(&g, xmin, xmax, ymin, ymax);
		for (i = 0; i < nsteps; i++)
			viz_grid_point(&g, elem(true_pos, i, 0), elem(true_pos, i, 1), '.');
		viz_grid_print(&g);
		printf("\n");
		viz_color(COL_DIM);
		printf("  trajectory preview -- press enter or wait...\n");
		viz_color(COL_RESET);
		usleep(2000000);
	}

	gettimeofday(&t_start, NULL);

	/* main filter loop */
	for (i = 0; i < nsteps; ) {
		float est_x, est_y, true_x, true_y, err, elapsed;
		float trace_p;
		int ret;

		if (!cfg->quiet && cfg->interactive) {
			while (paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) { action = -1; goto end_loop_2d; }
				if (ret == -2) { action = -2; goto end_loop_2d; }
				if (ret == -3) { action = -3; goto end_loop_2d; }
				if (ret == 1) break;
				usleep(20000);
			}
			if (!paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) { action = -1; goto end_loop_2d; }
				if (ret == -2) { action = -2; goto end_loop_2d; }
				if (ret == -3) { action = -3; goto end_loop_2d; }
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

		/* compute NIS if metrics enabled */
		nis = 0;
		if (cfg->metrics) {
			/* compute predicted measurement covariance: Cy = H * P * H^T + R
			 * for position-only measurement, H = [I 0] (2x6) */
			{
				Matrix H, HP, HPT, Cy;
				int r;

				/* H = [I 0] 2x6 matrix */
				H = zeroMatrix(2, 6);
				for (r = 0; r < 2; r++)
					setElem(H, r, r, 1.0);

				/* HP = H * P */
				HP = mulMatrix(H, CEst);

				/* HPT = HP * H^T = H * P * H^T */
				HPT = mulMatrix(HP, transposeMatrix(H));

				/* Cy = H * P * H^T + R = HPT + Cv */
				Cy = addMatrix(HPT, Cv);

				/* compute NIS */
				nis = compute_nis_2d(innov_x, innov_y, Cy);
				nis_sum += nis;

				/* save history (circular buffer) */
				innov_hist[innov_idx] = sqrt(innov_x*innov_x + innov_y*innov_y);
				rmse_hist[rmse_idx] = err_sum / (i + 1);
				innov_idx = (innov_idx + 1) % 50;
				rmse_idx = (rmse_idx + 1) % 50;

				/* cleanup */
				freeMatrix(H);
				freeMatrix(HP);
				freeMatrix(HPT);
				freeMatrix(Cy);
			}
		}

		/* update */
		gaussianEstimator_Est(&xEst, &CEst, &y, &Cv, hfun_2d, &m_opt);

		est_x = elem(xEst, 0, 0);
		est_y = elem(xEst, 1, 0);
		err = sqrt((est_x - true_x) * (est_x - true_x) +
		           (est_y - true_y) * (est_y - true_y));
		err_sum += err;
		if (err > err_max) err_max = err;

		trace_p = elem(CEst, 0, 0) + elem(CEst, 1, 1);

		/* export row if file is open */
		if (expf) {
			export_row_2d(expf, i, i * dt,
				true_x, true_y,
				elem(meas, i, 0), elem(meas, i, 1),
				est_x, est_y,
				elem(xEst, 2, 0), elem(xEst, 3, 0),
				elem(CEst, 0, 0), elem(CEst, 1, 1),
				err_sum / (i + 1), nis);
		}

		if (!cfg->quiet) {
			/* draw grid */
			viz_grid_init(&g, xmin, xmax, ymin, ymax);

			/* draw covariance ellipse first (so markers draw on top) */
			viz_grid_ellipse(&g, est_x, est_y, CEst, '~');

			/* plot true trajectory with trail effect */
			{
				int k, trail = 10;
				for (k = 0; k <= i; k++) {
					char ch = (i - k > trail) ? ',' : '.';
					viz_grid_point(&g, elem(true_pos, k, 0), elem(true_pos, k, 1), ch);
				}
			}

			/* plot measurements — current one highlighted */
			{
				int k;
				for (k = 0; k < i; k++)
					viz_grid_point(&g, elem(meas, k, 0), elem(meas, k, 1), 'o');
				/* current measurement: bright marker */
				viz_grid_point(&g, elem(meas, i, 0), elem(meas, i, 1), 'O');
			}

			/* plot estimate */
			viz_grid_point(&g, est_x, est_y, '+');

			gettimeofday(&t_now, NULL);
			elapsed = (t_now.tv_sec - t_start.tv_sec) +
				(t_now.tv_usec - t_start.tv_usec) / 1e6;
			render_frame_2d(&g, cfg, i, nsteps,
				true_x, true_y, elem(meas, i, 0), elem(meas, i, 1),
				est_x, est_y, elem(xEst, 2, 0), elem(xEst, 3, 0),
				elem(CEst, 0, 0), elem(CEst, 1, 1), trace_p, trace_p0,
				err_sum / (i + 1), err, innov_x, innov_y, nis,
				nis_sum / (i + 1), paused, elapsed,
				innov_hist, rmse_hist, innov_idx);

			/* brief pause after measurement update for visual emphasis */
			usleep(speed * 1000);
		}

		i++;
	}

end_loop_2d:
	/* quiet mode: print text summary */
	if (cfg->quiet && action == 0) {
		printf("--- summary ---\n");
		printf("mean rmse: %.3f\n", err_sum / (i > 0 ? i : 1));
		printf("max error: %.3f\n", err_max);
		printf("final estimate: (%.3f, %.3f)\n",
			elem(xEst, 0, 0), elem(xEst, 1, 0));
		printf("final truth:    (%.3f, %.3f)\n",
			elem(true_pos, nsteps - 1, 0), elem(true_pos, nsteps - 1, 1));
	}

	/* show visual end-of-demo summary */
	if (!cfg->quiet && action == 0) {
		float final_rmse = err_sum / (i > 0 ? i : 1);
		float final_trace = elem(CEst, 0, 0) + elem(CEst, 1, 1);

		viz_clear_screen();
		viz_cursor_move(1, 1);
		viz_color(COL_BOLD);
		printf("  demo complete — %s\n\n", sim_trajectory_name(traj_type));
		viz_color(COL_RESET);
		printf("  steps:          %d\n", i);
		printf("  final RMSE:     ");
		if (final_rmse < 2.0) viz_color(COL_GREEN);
		else if (final_rmse < 5.0) viz_color(COL_YELLOW);
		else viz_color(COL_RED);
		printf("%.3f\n", final_rmse);
		viz_color(COL_RESET);
		printf("  avg RMSE:       %.3f\n", final_rmse);
		printf("  max error:      %.3f\n", err_max);
		printf("  final trace(P): %.3f\n", final_trace);
		printf("  final estimate: (%.2f, %.2f)\n",
			elem(xEst, 0, 0), elem(xEst, 1, 0));
		printf("  final truth:    (%.2f, %.2f)\n",
			elem(true_pos, nsteps - 1, 0), elem(true_pos, nsteps - 1, 1));

		if (cfg->interactive || cfg->loop) {
			printf("\n");
			viz_color(COL_DIM);
			printf("  [r] restart  [n] next trajectory  [q] quit\n");
			viz_color(COL_RESET);

			if (cfg->loop) {
				/* auto-restart after a short pause */
				usleep(1000000);
				action = -2;
			} else {
				/* wait for user input */
				while (1) {
					if (term_kbhit()) {
						int ch = term_getchar();
						if (ch == 'q' || ch == 'Q') { action = -1; break; }
						if (ch == 'r' || ch == 'R') { action = -2; break; }
						if (ch == 'n' || ch == 'N') { action = -3; break; }
						if (ch == ' ' || ch == '\n') { action = -1; break; }
					}
					usleep(20000);
				}
			}
		}
	}

	/* free per-run state */
	freeMatrix(xEst);
	freeMatrix(CEst);
	sim_free_scenario(scen);

	/* handle restart / next trajectory */
	if (action == -2) {
		srand(time(NULL));
		goto restart_2d;
	}
	if (action == -3) {
		traj_type = (traj_type + 1) % 4;
		srand(time(NULL));
		goto restart_2d;
	}

	if (!cfg->quiet && cfg->interactive)
		term_restore();

	if (action != 0) {
		/* early quit — print short summary */
		printf("\n--- summary ---\n");
		printf("stopped at step %d/%d\n", i, nsteps);
	}

	if (expf)
		export_close(expf);

cleanup_2d:
	freeMatrix(Cw);
	freeMatrix(Cv);
	freeMatrix(m_opt);
	freeMatrix(y);
}

/* multi-target rendering */
static void render_frame_multi(Grid *g, Config *cfg, Target *targets, int ntargets,
	int step, int nsteps, float avg_rmse,
	int paused, float elapsed) {
	int k, wide;
	Panel sp;
	char buf[64];
	const char *target_cols[] = {COL_GREEN, COL_YELLOW, COL_CYAN, COL_RED};

	wide = viz_term_width() >= 80;

	viz_clear_screen();

	/* header */
	viz_cursor_move(1, 1);
	printf("Multi-target tracking (%d targets)  ", ntargets);
	viz_color(COL_BOLD);
	printf("[step %d/%d]", step + 1, nsteps);
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
	viz_color(COL_DIM);
	printf("  %.1fs", elapsed);
	viz_color(COL_RESET);

	/* grid */
	viz_grid_print_at(g, 3, 1);

	if (wide) {
		sp.row = 3;
		sp.col = 7 + GRID_W + 2;
		sp.width = 18;
		sp.height = 4 + ntargets * 3;
		viz_panel_border(&sp);

		snprintf(buf, sizeof(buf), " %d targets", ntargets);
		viz_panel_text(&sp, 0, buf);

		for (k = 0; k < ntargets; k++) {
			int line = 2 + k * 3;
			float ex = elem(targets[k].xEst, 0, 0);
			float ey = elem(targets[k].xEst, 1, 0);

			viz_cursor_move(sp.row + 1 + line, sp.col + 1);
			viz_color(target_cols[k]);
			printf(" %c", targets[k].marker);
			viz_color(COL_RESET);
			printf(": %5.1f,%5.1f", ex, ey);

			snprintf(buf, sizeof(buf), "   P:%.1f",
				elem(targets[k].CEst, 0, 0) + elem(targets[k].CEst, 1, 1));
			viz_panel_text(&sp, line + 1, buf);
		}
	}

	/* legend below grid */
	{
		int brow = 3 + GRID_H + 2;
		viz_cursor_move(brow, 1);

		for (k = 0; k < ntargets; k++) {
			viz_cursor_move(brow + k, 1);
			printf("  ");
			viz_color(target_cols[k]);
			printf("%c", targets[k].marker);
			viz_color(COL_RESET);
			printf("/%c target %d  ", targets[k].marker + 32, k + 1);
		}

		viz_cursor_move(brow + ntargets, 1);
		printf("  ");
		viz_color(COL_DIM);
		printf("?");
		viz_color(COL_RESET);
		printf(" measurement dropout");

		viz_cursor_move(brow + ntargets + 1, 1);
		printf("  avg RMSE: ");
		if (avg_rmse < 2.0) viz_color(COL_GREEN);
		else if (avg_rmse < 5.0) viz_color(COL_YELLOW);
		else viz_color(COL_RED);
		printf("%.3f", avg_rmse);
		viz_color(COL_RESET);

		viz_cursor_move(brow + ntargets + 2, 3);
		viz_progress_bar(step, nsteps, 30);
	}

	if (step == 0 && cfg->interactive) {
		int hrow = 3 + GRID_H + ntargets + 5;
		viz_cursor_move(hrow, 1);
		viz_color(COL_DIM);
		printf("  [space] step  [p]ause  [r]estart  [+/-] speed  [q]uit");
		viz_color(COL_RESET);
	}

	viz_cursor_move(3 + GRID_H + ntargets + 7, 1);
	fflush(stdout);
}

static void run_demo_multi(Config *cfg) {
	int i, k;
	int nsteps = cfg->nsteps;
	float dt = cfg->dt;
	int L = cfg->L;
	int ntargets = cfg->ntargets;
	float xmin, xmax, ymin, ymax, margin;
	float err_sum[MAX_TARGETS] = {0};
	float err_max[MAX_TARGETS] = {0};
	float avg_rmse;
	int paused, speed;
	struct timeval t_start, t_now;
	int action = 0;

	Target targets[MAX_TARGETS];
	Matrix Cw, Cv, m_opt, y;
	Grid g;

	if (nsteps <= 0) {
		printf("nothing to do (nsteps=0)\n");
		return;
	}
	if (ntargets < 1) ntargets = 1;
	if (ntargets > MAX_TARGETS) ntargets = MAX_TARGETS;

	/* shared filter constants — same for all targets */
	Cw = zeroMatrix(6, 6);
	setElem(Cw, 0, 0, 0.01);
	setElem(Cw, 1, 1, 0.01);
	setElem(Cw, 2, 2, 0.1);
	setElem(Cw, 3, 3, 0.1);
	setElem(Cw, 4, 4, 0.001);
	setElem(Cw, 5, 5, 0.001);

	Cv = zeroMatrix(2, 2);
	setElem(Cv, 0, 0, 4.0);
	setElem(Cv, 1, 1, 4.0);

	m_opt = gaussianApprox(L);
	y = newMatrix(2, 1);

	speed = cfg->speed;

	if (!cfg->quiet && cfg->interactive)
		term_raw_mode();

	/* generate scenarios for all targets */
	sim_multi_scenario(targets, ntargets, nsteps, dt, 2.0);

	/* init filter state for each target */
	for (k = 0; k < ntargets; k++) {
		if (!targets[k].scen) continue;
		targets[k].xEst = zeroMatrix(6, 1);
		setElem(targets[k].xEst, 0, 0, elem(targets[k].scen->true_pos, 0, 0));
		setElem(targets[k].xEst, 1, 0, elem(targets[k].scen->true_pos, 0, 1));

		targets[k].CEst = zeroMatrix(6, 6);
		setElem(targets[k].CEst, 0, 0, 10.0);
		setElem(targets[k].CEst, 1, 1, 10.0);
		setElem(targets[k].CEst, 2, 2, 5.0);
		setElem(targets[k].CEst, 3, 3, 5.0);
		setElem(targets[k].CEst, 4, 4, 0.001);
		setElem(targets[k].CEst, 5, 5, 0.001);
	}

	/* auto-scale grid to fit all targets */
	xmin = xmax = elem(targets[0].scen->true_pos, 0, 0);
	ymin = ymax = elem(targets[0].scen->true_pos, 0, 1);
	for (k = 0; k < ntargets; k++) {
		Matrix pos = targets[k].scen->true_pos;
		for (i = 0; i < nsteps; i++) {
			float tx = elem(pos, i, 0);
			float ty = elem(pos, i, 1);
			if (tx < xmin) xmin = tx;
			if (tx > xmax) xmax = tx;
			if (ty < ymin) ymin = ty;
			if (ty > ymax) ymax = ty;
		}
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

	paused = cfg->interactive;

	if (!cfg->quiet) {
		viz_clear_screen();
		printf("Multi-target tracking demo\n");
		printf("targets: %d, dt=%.2f, L=%d, nsteps=%d\n\n", ntargets, dt, L, nsteps);

		/* trajectory preview */
		viz_grid_init(&g, xmin, xmax, ymin, ymax);
		for (k = 0; k < ntargets; k++) {
			Matrix pos = targets[k].scen->true_pos;
			for (i = 0; i < nsteps; i++)
				viz_grid_point(&g, elem(pos, i, 0), elem(pos, i, 1), targets[k].marker + 32);
		}
		viz_grid_print(&g);
		printf("\n");
		viz_color(COL_DIM);
		printf("  trajectory preview -- press enter or wait...\n");
		viz_color(COL_RESET);
		usleep(2000000);
	}

	gettimeofday(&t_start, NULL);

	for (i = 0; i < nsteps; ) {
		float elapsed;
		int ret;

		if (!cfg->quiet && cfg->interactive) {
			while (paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) { action = -1; goto end_multi; }
				if (ret == -2) { action = -2; goto end_multi; }
				if (ret == 1) break;
				usleep(20000);
			}
			if (!paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) { action = -1; goto end_multi; }
				if (ret == -2) { action = -2; goto end_multi; }
			}
		}

		/* run filter for each target */
		for (k = 0; k < ntargets; k++) {
			float est_x, est_y, true_x, true_y, err;
			int dropout;
			Matrix meas;

			if (!targets[k].active) continue;

			true_x = elem(targets[k].scen->true_pos, i, 0);
			true_y = elem(targets[k].scen->true_pos, i, 1);
			meas = targets[k].scen->measurements;

			/* predict */
			gaussianEstimator_Pred(&targets[k].xEst, &targets[k].CEst,
				NULL, &Cw, afun_2d, &dt, &m_opt);

			/* measurement dropout check */
			dropout = sim_measurement_dropout();

			if (!dropout) {
				setElem(y, 0, 0, elem(meas, i, 0));
				setElem(y, 1, 0, elem(meas, i, 1));
				/* update with measurement */
				gaussianEstimator_Est(&targets[k].xEst, &targets[k].CEst,
					&y, &Cv, hfun_2d, &m_opt);
			}
			/* if dropout, skip update — covariance grows from prediction only */

			est_x = elem(targets[k].xEst, 0, 0);
			est_y = elem(targets[k].xEst, 1, 0);
			err = sqrt((est_x - true_x) * (est_x - true_x) +
			           (est_y - true_y) * (est_y - true_y));
			err_sum[k] += err;
			if (err > err_max[k]) err_max[k] = err;
		}

		/* compute average rmse across all targets */
		avg_rmse = 0;
		for (k = 0; k < ntargets; k++)
			avg_rmse += err_sum[k] / (i + 1);
		avg_rmse /= ntargets;

		if (!cfg->quiet) {
			/* render grid */
			viz_grid_init(&g, xmin, xmax, ymin, ymax);

			/* draw ellipses first */
			for (k = 0; k < ntargets; k++) {
				float ex = elem(targets[k].xEst, 0, 0);
				float ey = elem(targets[k].xEst, 1, 0);
				viz_grid_ellipse(&g, ex, ey, targets[k].CEst, '~');
			}

			/* draw trajectories with trail */
			for (k = 0; k < ntargets; k++) {
				Matrix pos = targets[k].scen->true_pos;
				Matrix ms = targets[k].scen->measurements;
				int j, trail = 10;
				char mch = targets[k].marker + 32;  /* lowercase */

				for (j = 0; j <= i; j++) {
					char ch = (i - j > trail) ? ',' : '.';
					viz_grid_point(&g, elem(pos, j, 0), elem(pos, j, 1), ch);
				}

				/* measurements */
				for (j = 0; j < i; j++)
					viz_grid_point(&g, elem(ms, j, 0), elem(ms, j, 1), mch);

				/* current measurement or dropout marker */
				if (!sim_measurement_dropout())
					viz_grid_point(&g, elem(ms, i, 0), elem(ms, i, 1), mch);
				else
					viz_grid_point(&g, elem(pos, i, 0), elem(pos, i, 1), '?');

				/* estimate marker */
				viz_grid_point(&g, elem(targets[k].xEst, 0, 0),
					elem(targets[k].xEst, 1, 0), targets[k].marker);
			}

			gettimeofday(&t_now, NULL);
			elapsed = (t_now.tv_sec - t_start.tv_sec) +
				(t_now.tv_usec - t_start.tv_usec) / 1e6;

			render_frame_multi(&g, cfg, targets, ntargets, i, nsteps,
				avg_rmse, paused, elapsed);

			usleep(speed * 1000);
		}

		i++;
	}

end_multi:
	if (!cfg->quiet && cfg->interactive)
		term_restore();

	/* print summary */
	{
		int worst = 0;
		if (action != 0)
			printf("\nstopped at step %d/%d\n", i, nsteps);
		printf("\n--- multi-target summary ---\n");
		printf("targets: %d, steps: %d\n", ntargets, i);
		for (k = 0; k < ntargets; k++) {
			float rmse_k = err_sum[k] / (i > 0 ? i : 1);
			printf("  target %c: avg RMSE=%.3f  max err=%.3f\n",
				targets[k].marker, rmse_k, err_max[k]);
			if (rmse_k > err_sum[worst] / (i > 0 ? i : 1)) worst = k;
		}
		printf("  worst: target %c\n", targets[worst].marker);
		printf("  overall avg RMSE: %.3f\n", avg_rmse);
	}

	/* cleanup */
	sim_free_targets(targets, ntargets);
	freeMatrix(Cw);
	freeMatrix(Cv);
	freeMatrix(m_opt);
	freeMatrix(y);
}

/* rotation demo — render frame */
static void render_frame_rot(Grid *g, Config *cfg, int step, int nsteps,
	float true_rot[3], float est_rot[3], float rot_err,
	float pos_err, float rmse, int paused, float elapsed) {
	int wide = viz_term_width() >= 80;
	Panel sp;
	char buf[64];

	viz_clear_screen();

	viz_cursor_move(1, 1);
	printf("3D rotation tracking  ");
	viz_color(COL_BOLD);
	printf("[step %d/%d]", step + 1, nsteps);
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
	viz_color(COL_DIM);
	printf("  %.1fs", elapsed);
	viz_color(COL_RESET);

	viz_grid_print_at(g, 3, 1);

	if (wide) {
		sp.row = 3;
		sp.col = 7 + GRID_W + 2;
		sp.width = 22;
		sp.height = 18;
		viz_panel_border(&sp);

		viz_panel_text(&sp, 0, "  rotation demo");

		snprintf(buf, sizeof(buf), " Step %d/%d", step + 1, nsteps);
		viz_panel_text(&sp, 2, buf);

		viz_cursor_move(sp.row + 1 + 4, sp.col + 1);
		viz_color(COL_GREEN);
		printf(" true rot:");
		viz_color(COL_RESET);
		snprintf(buf, sizeof(buf), "  %.2f %.2f %.2f", true_rot[0], true_rot[1], true_rot[2]);
		viz_panel_text(&sp, 5, buf);

		viz_cursor_move(sp.row + 1 + 7, sp.col + 1);
		viz_color(COL_CYAN);
		printf(" est rot:");
		viz_color(COL_RESET);
		snprintf(buf, sizeof(buf), "  %.2f %.2f %.2f", est_rot[0], est_rot[1], est_rot[2]);
		viz_panel_text(&sp, 8, buf);

		snprintf(buf, sizeof(buf), " rot err: %.3f", rot_err);
		viz_panel_text(&sp, 10, buf);
		snprintf(buf, sizeof(buf), " pos err: %.3f", pos_err);
		viz_panel_text(&sp, 11, buf);

		viz_cursor_move(sp.row + 1 + 13, sp.col + 1);
		printf(" RMSE: ");
		if (rmse < 2.0) viz_color(COL_GREEN);
		else if (rmse < 5.0) viz_color(COL_YELLOW);
		else viz_color(COL_RED);
		printf("%5.2f", rmse);
		viz_color(COL_RESET);
	}

	/* legend below grid */
	{
		int brow = 3 + GRID_H + 2;
		viz_cursor_move(brow, 1);
		printf("  ");
		viz_color(COL_GREEN);
		printf(".");
		viz_color(COL_RESET);
		printf(" true position    ");
		viz_color(COL_CYAN);
		printf("+");
		viz_color(COL_RESET);
		printf(" estimate");

		viz_cursor_move(brow + 1, 1);
		printf("  ");
		viz_color(COL_RED);
		printf("x");
		viz_color(COL_RESET);
		printf("/");
		viz_color(COL_GREEN);
		printf("y");
		viz_color(COL_RESET);
		printf("/");
		viz_color(COL_BLUE);
		printf("z");
		viz_color(COL_RESET);
		printf(" rotated axes");

		if (!wide) {
			viz_cursor_move(brow + 2, 1);
			printf("  rot err: %.3f  pos err: %.3f  RMSE: ", rot_err, pos_err);
			if (rmse < 2.0) viz_color(COL_GREEN);
			else if (rmse < 5.0) viz_color(COL_YELLOW);
			else viz_color(COL_RED);
			printf("%.3f", rmse);
			viz_color(COL_RESET);
		}

		viz_cursor_move(brow + (wide ? 2 : 3), 3);
		viz_progress_bar(step, nsteps, 30);
	}

	if (step == 0 && cfg->interactive) {
		int hrow = 3 + GRID_H + (wide ? 5 : 6);
		viz_cursor_move(hrow, 1);
		viz_color(COL_DIM);
		printf("  [space] step  [p]ause  [r]estart  [+/-] speed  [q]uit");
		viz_color(COL_RESET);
	}

	viz_cursor_move(3 + GRID_H + (wide ? 7 : 8), 1);
	fflush(stdout);
}


/* compare mode: compare L=3, L=5, L=7 approximation levels */
static void run_demo_compare(Config *cfg) {
	int levels[] = {3, 5, 7};
	int i;
	float dt = cfg->dt;
	int nsteps = cfg->nsteps;
	float err_sum[3] = {0, 0, 0};

	printf("\n========================================\n");
	printf("   Approximation Level Comparison\n");
	printf("========================================\n\n");
	printf("trajectory: %s\n", sim_trajectory_name(cfg->trajectory));
	printf("steps: %d, dt: %.2f\n\n", nsteps, dt);

	/* run filter for each L value with same scenario */
	for (i = 0; i < 3; i++) {
		int L = levels[i];
		int step;
		Matrix xEst, CEst, Cw, Cv, m_opt, y;
		Matrix true_pos, meas;
		float err_sum_i = 0;

		/* generate scenario once and reuse */
		if (i == 0) {
			Scenario *scen = sim_create_scenario(cfg->trajectory, nsteps, dt, 2.0);
			true_pos = scen->true_pos;
			meas = scen->measurements;
		} else {
			/* re-use same trajectory with new seed */
			Scenario *scen = sim_create_scenario(cfg->trajectory, nsteps, dt, 2.0);
			true_pos = scen->true_pos;
			meas = scen->measurements;
		}

		/* initial state estimate */
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

		/* filter loop */
		for (step = 0; step < nsteps; step++) {
			float est_x, est_y, true_x, true_y, err;

			/* measurement */
			setElem(y, 0, 0, elem(meas, step, 0));
			setElem(y, 1, 0, elem(meas, step, 1));

			/* predict */
			gaussianEstimator_Pred(&xEst, &CEst, NULL, &Cw, afun_2d, &dt, &m_opt);

			/* update */
			gaussianEstimator_Est(&xEst, &CEst, &y, &Cv, hfun_2d, &m_opt);

			/* compute error */
			est_x = elem(xEst, 0, 0);
			est_y = elem(xEst, 1, 0);
			true_x = elem(true_pos, step, 0);
			true_y = elem(true_pos, step, 1);
			err = sqrt((est_x - true_x) * (est_x - true_x) +
			           (est_y - true_y) * (est_y - true_y));
			err_sum_i += err;
		}

		freeMatrix(xEst);
		freeMatrix(CEst);
		freeMatrix(Cw);
		freeMatrix(Cv);
		freeMatrix(m_opt);
		freeMatrix(y);
		
		err_sum[i] = err_sum_i;
	}

	/* print comparison results */
	printf("\nL=3 (2 points): RMSE=%.4f\n", err_sum[0] / nsteps);
	printf("L=5 (4 points): RMSE=%.4f\n", err_sum[1] / nsteps);
	printf("L=7 (6 points): RMSE=%.4f\n\n", err_sum[2] / nsteps);

	printf("\n----------------------------------------\n");

	/* find best */
	{
		float best_err = err_sum[0] / nsteps;
		int best_i = 0;
		for (i = 1; i < 3; i++) {
			float rmse = err_sum[i] / nsteps;
			if (rmse < best_err) {
				best_err = rmse;
				best_i = i;
			}
		}
		printf("recommendation: L=%d gives best accuracy (RMSE=%.4f)\n",
		       levels[best_i], best_err);
	}
	printf("========================================\n");
}


static void run_demo_rot(Config *cfg) {
	int i;
	int nsteps = cfg->nsteps;
	float dt = cfg->dt;
	int L = cfg->L;
	float err_sum = 0;
	int paused, speed;
	struct timeval t_start, t_now;
	int action = 0;

	/* true state: [pos(3), rot(3), angvel(3), linvel(3)] = 12 elements */
	float true_pos[3] = {0, 0, 0};
	float true_rot[3] = {0, 0, 0};
	float true_angvel[3] = {0, 0, 0.3};  /* slow z rotation */
	float true_linvel[3] = {0.5, 0.2, 0};

	Matrix xEst, CEst, Cw, Cv, m_opt, y;
	Grid g;
	float xmin, xmax, ymin, ymax;

	if (nsteps <= 0) {
		printf("nothing to do (nsteps=0)\n");
		return;
	}

	/* initial filter state — 12-state vector */
	xEst = zeroMatrix(12, 1);
	setElem(xEst, 0, 0, 0.0);  /* pos x */
	setElem(xEst, 1, 0, 0.0);  /* pos y */
	setElem(xEst, 2, 0, 0.0);  /* pos z */
	/* rot, angvel, linvel start at zero */

	/* initial covariance 12x12 */
	CEst = zeroMatrix(12, 12);
	/* position uncertainty */
	setElem(CEst, 0, 0, 5.0);
	setElem(CEst, 1, 1, 5.0);
	setElem(CEst, 2, 2, 5.0);
	/* high rotation uncertainty */
	setElem(CEst, 3, 3, 1.0);
	setElem(CEst, 4, 4, 1.0);
	setElem(CEst, 5, 5, 1.0);
	/* angular velocity */
	setElem(CEst, 6, 6, 0.5);
	setElem(CEst, 7, 7, 0.5);
	setElem(CEst, 8, 8, 0.5);
	/* linear velocity */
	setElem(CEst, 9, 9, 2.0);
	setElem(CEst, 10, 10, 2.0);
	setElem(CEst, 11, 11, 2.0);

	/* process noise 12x12 */
	Cw = zeroMatrix(12, 12);
	setElem(Cw, 0, 0, 0.01);
	setElem(Cw, 1, 1, 0.01);
	setElem(Cw, 2, 2, 0.01);
	setElem(Cw, 3, 3, 0.005);
	setElem(Cw, 4, 4, 0.005);
	setElem(Cw, 5, 5, 0.005);
	setElem(Cw, 6, 6, 0.01);
	setElem(Cw, 7, 7, 0.01);
	setElem(Cw, 8, 8, 0.01);
	setElem(Cw, 9, 9, 0.05);
	setElem(Cw, 10, 10, 0.05);
	setElem(Cw, 11, 11, 0.05);

	/* measurement noise — observe position(3) */
	Cv = zeroMatrix(3, 3);
	setElem(Cv, 0, 0, 4.0);
	setElem(Cv, 1, 1, 4.0);
	setElem(Cv, 2, 2, 4.0);

	m_opt = gaussianApprox(L);
	y = newMatrix(3, 1);

	speed = cfg->speed;
	paused = cfg->interactive;

	if (!cfg->quiet && cfg->interactive)
		term_raw_mode();

	/* pre-compute trajectory for bounds */
	{
		float px = 0, py = 0, pz = 0;
		xmin = xmax = 0;
		ymin = ymax = 0;
		for (i = 0; i < nsteps; i++) {
			float ppx, ppy;
			px += true_linvel[0] * dt;
			py += true_linvel[1] * dt;
			pz += 0;  /* no z drift */
			viz_project_3d(px, py, pz, &ppx, &ppy);
			ppx *= 2.0;
			if (ppx - 5 < xmin) xmin = ppx - 5;
			if (ppx + 5 > xmax) xmax = ppx + 5;
			if (ppy - 4 < ymin) ymin = ppy - 4;
			if (ppy + 4 > ymax) ymax = ppy + 4;
		}
	}
	if (xmax - xmin < 10) { xmin -= 5; xmax += 5; }
	if (ymax - ymin < 8) { ymin -= 4; ymax += 4; }

	if (!cfg->quiet) {
		viz_clear_screen();
		printf("3D rotation tracking demo\n");
		printf("12-state: [pos, rot, angvel, linvel]\n");
		printf("dt=%.2f, L=%d, nsteps=%d\n\n", dt, L, nsteps);
		usleep(1500000);
	}

	gettimeofday(&t_start, NULL);

restart_rot:
	action = 0;
	err_sum = 0;

	for (i = 0; i < nsteps; ) {
		float est_pos[3], est_rot[3];
		float pos_err, rot_err, elapsed, rmse;
		int ret;

		if (!cfg->quiet && cfg->interactive) {
			while (paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) { action = -1; goto end_rot; }
				if (ret == -2) { action = -2; goto end_rot; }
				if (ret == 1) break;
				usleep(20000);
			}
			if (!paused) {
				ret = handle_input(&paused, &speed);
				if (ret == -1) { action = -1; goto end_rot; }
				if (ret == -2) { action = -2; goto end_rot; }
			}
		}

		/* propagate true state */
		true_pos[0] += true_linvel[0] * dt + 0.01 * randn();
		true_pos[1] += true_linvel[1] * dt + 0.01 * randn();
		true_pos[2] += 0.01 * randn();  /* slight z wobble */

		/* rotation: add angular velocity * dt + noise */
		true_rot[0] += (true_angvel[0] + 0.05 * randn()) * dt;
		true_rot[1] += (true_angvel[1] + 0.05 * randn()) * dt;
		true_rot[2] += (true_angvel[2]) * dt;

		/* slight wobble on x angular velocity */
		true_angvel[0] = 0.1 * sin(0.02 * i);

		/* generate noisy position measurement */
		setElem(y, 0, 0, true_pos[0] + 2.0 * randn());
		setElem(y, 1, 0, true_pos[1] + 2.0 * randn());
		setElem(y, 2, 0, true_pos[2] + 2.0 * randn());

		/* predict using decomp (rotation-aware) */
		gaussianEstimator_Pred_decomp(&xEst, &CEst, NULL, &Cw, &dt, &m_opt);

		/* update with position measurement */
		gaussianEstimator_Est(&xEst, &CEst, &y, &Cv, hfun_3d, &m_opt);

		/* extract estimates */
		est_pos[0] = elem(xEst, 0, 0);
		est_pos[1] = elem(xEst, 1, 0);
		est_pos[2] = elem(xEst, 2, 0);
		est_rot[0] = elem(xEst, 3, 0);
		est_rot[1] = elem(xEst, 4, 0);
		est_rot[2] = elem(xEst, 5, 0);

		/* position error */
		pos_err = sqrt((est_pos[0] - true_pos[0]) * (est_pos[0] - true_pos[0]) +
		               (est_pos[1] - true_pos[1]) * (est_pos[1] - true_pos[1]) +
		               (est_pos[2] - true_pos[2]) * (est_pos[2] - true_pos[2]));

		/* rotation error: angle between rotation vectors */
		{
			float dr[3];
			dr[0] = est_rot[0] - true_rot[0];
			dr[1] = est_rot[1] - true_rot[1];
			dr[2] = est_rot[2] - true_rot[2];
			rot_err = sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2]);
		}

		err_sum += pos_err;
		rmse = err_sum / (i + 1);

		if (!cfg->quiet) {
			float px_true, py_true, px_est, py_est;

			viz_grid_init(&g, xmin, xmax, ymin, ymax);

			/* draw true axes (green-ish) */
			{
				float proj_cx, proj_cy;
				viz_project_3d(true_pos[0], true_pos[1], true_pos[2],
					&proj_cx, &proj_cy);
				proj_cx *= 2.0;
				viz_grid_axes_3d(&g, proj_cx, proj_cy, true_rot);
			}

			/* draw position trail */
			{
				/* just plot current true + estimated positions */
				viz_project_3d(true_pos[0], true_pos[1], true_pos[2],
					&px_true, &py_true);
				px_true *= 2.0;
				viz_grid_point(&g, px_true, py_true, '.');

				viz_project_3d(est_pos[0], est_pos[1], est_pos[2],
					&px_est, &py_est);
				px_est *= 2.0;
				viz_grid_point(&g, px_est, py_est, '+');
			}

			/* draw estimated axes in different shade */
			{
				float proj_cx, proj_cy;
				viz_project_3d(est_pos[0], est_pos[1], est_pos[2],
					&proj_cx, &proj_cy);
				proj_cx *= 2.0;
				viz_grid_axes_3d(&g, proj_cx, proj_cy, est_rot);
			}

			/* measurement marker */
			{
				float mx, my;
				viz_project_3d(elem(y, 0, 0), elem(y, 1, 0), elem(y, 2, 0),
					&mx, &my);
				mx *= 2.0;
				viz_grid_point(&g, mx, my, 'o');
			}

			gettimeofday(&t_now, NULL);
			elapsed = (t_now.tv_sec - t_start.tv_sec) +
				(t_now.tv_usec - t_start.tv_usec) / 1e6;

			render_frame_rot(&g, cfg, i, nsteps,
				true_rot, est_rot, rot_err, pos_err, rmse,
				paused, elapsed);

			usleep(speed * 1000);
		}

		i++;
	}

end_rot:
	if (!cfg->quiet && cfg->interactive)
		term_restore();

	/* summary */
	if (action == 0) {
		float final_rmse = err_sum / (i > 0 ? i : 1);
		printf("\n--- rotation demo summary ---\n");
		printf("steps: %d\n", i);
		printf("avg pos RMSE: %.3f\n", final_rmse);
		printf("final pos estimate: (%.2f, %.2f, %.2f)\n",
			elem(xEst, 0, 0), elem(xEst, 1, 0), elem(xEst, 2, 0));
		printf("final rot estimate: (%.3f, %.3f, %.3f)\n",
			elem(xEst, 3, 0), elem(xEst, 4, 0), elem(xEst, 5, 0));
		printf("true rot: (%.3f, %.3f, %.3f)\n",
			true_rot[0], true_rot[1], true_rot[2]);
	} else {
		printf("\nstopped at step %d/%d\n", i, nsteps);
	}

	if (action == -2) {
		srand(time(NULL));
		true_pos[0] = true_pos[1] = true_pos[2] = 0;
		true_rot[0] = true_rot[1] = true_rot[2] = 0;
		freeMatrix(xEst);
		freeMatrix(CEst);
		xEst = zeroMatrix(12, 1);
		CEst = zeroMatrix(12, 12);
		setElem(CEst, 0, 0, 5.0);
		setElem(CEst, 1, 1, 5.0);
		setElem(CEst, 2, 2, 5.0);
		setElem(CEst, 3, 3, 1.0);
		setElem(CEst, 4, 4, 1.0);
		setElem(CEst, 5, 5, 1.0);
		setElem(CEst, 6, 6, 0.5);
		setElem(CEst, 7, 7, 0.5);
		setElem(CEst, 8, 8, 0.5);
		setElem(CEst, 9, 9, 2.0);
		setElem(CEst, 10, 10, 2.0);
		setElem(CEst, 11, 11, 2.0);
		goto restart_rot;
	}

	freeMatrix(xEst);
	freeMatrix(CEst);
	freeMatrix(Cw);
	freeMatrix(Cv);
	freeMatrix(m_opt);
	freeMatrix(y);
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
	if (strcmp(s, "multi") == 0) return MODE_MULTI;
	if (strcmp(s, "rot") == 0) return MODE_ROT;
	if (strcmp(s, "test") == 0) return MODE_TEST;
	if (strcmp(s, "compare") == 0) return MODE_COMPARE;
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
	cfg.loop = 0;
	cfg.outfile = NULL;
	cfg.ntargets = 2;
	cfg.verbose = 0;
	cfg.metrics = 0;

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
		} else if (strcmp(argv[i], "--loop") == 0) {
			cfg.loop = 1;
			memmove(&argv[i], &argv[i + 1], (argc - i - 1) * sizeof(char *));
			argc--;
			i--;
		} else if (strcmp(argv[i], "--metrics") == 0) {
			cfg.metrics = 1;
			memmove(&argv[i], &argv[i + 1], (argc - i - 1) * sizeof(char *));
			argc--;
			i--;
		}
	}

	while ((opt = getopt(argc, argv, "m:t:n:d:L:s:o:k:qivh")) != -1) {
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
			break;
		case 's':
			cfg.seed = atoi(optarg);
			break;
		case 'k':
			cfg.ntargets = atoi(optarg);
			if (cfg.ntargets < 1 || cfg.ntargets > MAX_TARGETS) {
				fprintf(stderr, "targets must be 1-%d\n", MAX_TARGETS);
				return 1;
			}
			break;
		case 'o':
			cfg.outfile = optarg;
			break;
		case 'q':
			cfg.quiet = 1;
			break;
		case 'i':
			cfg.interactive = 1;
			break;
		case 'v':
			cfg.verbose = 1;
			break;
		case 'h':
			print_usage(argv[0]);
			return 0;
		default:
			print_usage(argv[0]);
			return 1;
		}
	}

	/* validate L */
	if (cfg.L != 3 && cfg.L != 5 && cfg.L != 7) {
		fprintf(stderr, "warning: L=%d not supported, using L=7\n", cfg.L);
		cfg.L = 7;
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
		const char *mnames[] = {"2d", "1d", "test", "grid", "multi", "rot", "compare"};
		printf("vizga: mode=%s traj=%s steps=%d dt=%.2f L=%d seed=%s color=%s%s speed=%dms%s%s%s",
			mnames[cfg.mode], sim_trajectory_name(cfg.trajectory),
			cfg.nsteps, cfg.dt, cfg.L,
			cfg.seed >= 0 ? "fixed" : "time",
			viz_color_enabled ? "on" : "off",
			cfg.interactive ? " interactive" : "",
			cfg.speed,
			cfg.loop ? " loop" : "",
			cfg.outfile ? " export=" : "",
			cfg.outfile ? cfg.outfile : "");
		if (cfg.mode == MODE_MULTI)
			printf(" targets=%d", cfg.ntargets);
		printf("\n");
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
	case MODE_MULTI:
		run_demo_multi(&cfg);
		break;
	case MODE_ROT:
		run_demo_rot(&cfg);
		break;
	case MODE_COMPARE:
		run_demo_compare(&cfg);
		break;
	case MODE_2D:
	default:
		run_demo_2d(&cfg);
		break;
	}

	return 0;
}

/* TODO: add measurement gating */
