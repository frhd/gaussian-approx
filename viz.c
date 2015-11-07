#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "viz.h"
#include "eig.h"
#include "tracker.h"

int viz_color_enabled = 1;

void viz_color(const char *code) {
	if (viz_color_enabled)
		printf("%s", code);
}

void viz_bar(float value, float max_val, int width) {
	int i, n, center;
	if (max_val <= 0) max_val = 1.0;

	if (value >= 0) {
		center = width / 2;
		n = (int)(value / max_val * (width / 2));
		if (n > width / 2) n = width / 2;
		for (i = 0; i < center; i++) printf(" ");
		for (i = 0; i < n; i++) printf("#");
		printf("\n");
	} else {
		center = width / 2;
		n = (int)(-value / max_val * (width / 2));
		if (n > width / 2) n = width / 2;
		for (i = 0; i < center - n; i++) printf(" ");
		for (i = 0; i < n; i++) printf("#");
		printf("\n");
	}
}

void viz_vector(Matrix v) {
	int i;
	float max_val = 0;

	/* find max absolute value */
	for (i = 0; i < v->height; i++) {
		float val = fabs(elem(v, i, 0));
		if (val > max_val) max_val = val;
	}
	if (max_val == 0) max_val = 1.0;

	/* scale markers */
	printf("%-30s%30s\n", "", "");
	printf("%-30.2f%30.2f\n", -max_val, max_val);

	for (i = 0; i < v->height; i++) {
		printf("[%2d] ", i);
		viz_bar(elem(v, i, 0), max_val, 54);
	}
}

void viz_gaussian_1d(float mean, float sigma, int width, int height) {
	int x, y;
	float xmin, xmax, dx, ymax;
	float *pdf;

	xmin = mean - 3 * sigma;
	xmax = mean + 3 * sigma;
	dx = (xmax - xmin) / (width - 1);
	ymax = 1.0 / (sigma * sqrt(2 * pi));

	pdf = (float *)malloc(width * sizeof(float));
	for (x = 0; x < width; x++) {
		float xv = xmin + x * dx;
		float z = (xv - mean) / sigma;
		pdf[x] = exp(-0.5 * z * z) / (sigma * sqrt(2 * pi));
	}

	for (y = height - 1; y >= 0; y--) {
		float ythresh = ymax * y / (height - 1);
		for (x = 0; x < width; x++) {
			if (pdf[x] >= ythresh) {
				viz_color(COL_BLUE);
				printf("*");
				viz_color(COL_RESET);
			} else
				printf(" ");
		}
		printf("\n");
	}

	/* x axis */
	viz_color(COL_DIM);
	for (x = 0; x < width; x++) printf("-");
	printf("\n");

	/* labels */
	printf("%-*.*f", width / 2, 2, xmin);
	printf("%*.2f", width - width / 2, xmax);
	viz_color(COL_RESET);
	printf("\n");
	viz_color(COL_CYAN);
	printf("%*s%.2f\n", width / 2 - 2, "", mean);
	viz_color(COL_RESET);

	free(pdf);
}

void viz_sigma_points_1d(float mean, float sigma, Matrix m_opt, int width) {
	int x, i, npts;
	float xmin, xmax, dx;

	xmin = mean - 3 * sigma;
	xmax = mean + 3 * sigma;
	dx = (xmax - xmin) / (width - 1);
	npts = m_opt->width;

	/* marker line */
	for (x = 0; x < width; x++) {
		float xv = xmin + x * dx;
		int marked = 0;

		/* check mean */
		if (fabs(xv - mean) < dx * 0.5) {
			viz_color(COL_CYAN);
			printf("+");
			viz_color(COL_RESET);
			marked = 1;
		}

		if (!marked) {
			/* check sigma points */
			for (i = 0; i < npts; i++) {
				float sp = mean + elem(m_opt, 0, i) * sigma;
				if (fabs(xv - sp) < dx * 0.5) {
					viz_color(COL_RED);
					printf("|");
					viz_color(COL_RESET);
					marked = 1;
					break;
				}
			}
		}

		if (!marked) printf(" ");
	}
	printf("\n");

	printf("(%d sigma points)\n", npts);
}

void viz_grid_init(Grid *g, float xmin, float xmax, float ymin, float ymax) {
	int r, c;

	g->xmin = xmin;
	g->xmax = xmax;
	g->ymin = ymin;
	g->ymax = ymax;

	for (r = 0; r < GRID_H; r++)
		for (c = 0; c < GRID_W; c++)
			g->cells[r][c] = ' ';

	for (c = 0; c < GRID_W; c++) {
		g->cells[0][c] = '-';
		g->cells[GRID_H - 1][c] = '-';
	}
	for (r = 0; r < GRID_H; r++) {
		g->cells[r][0] = '|';
		g->cells[r][GRID_W - 1] = '|';
	}
	g->cells[0][0] = '+';
	g->cells[0][GRID_W - 1] = '+';
	g->cells[GRID_H - 1][0] = '+';
	g->cells[GRID_H - 1][GRID_W - 1] = '+';
}

void viz_grid_print(Grid *g) {
	int r, c;
	char label[16];
	float yval, xmid;

	for (r = 0; r < GRID_H; r++) {
		yval = g->ymax - (g->ymax - g->ymin) * r / (GRID_H - 1);
		if (r == 0 || r == GRID_H / 2 || r == GRID_H - 1) {
			snprintf(label, sizeof(label), "%6.2f ", yval);
			printf("%s", label);
		} else {
			printf("       ");
		}
		for (c = 0; c < GRID_W; c++) {
			char ch = g->cells[r][c];
			if (ch == '.' ) {
				viz_color(COL_GREEN);
				putchar(ch);
				viz_color(COL_RESET);
			} else if (ch == 'o') {
				viz_color(COL_YELLOW);
				putchar(ch);
				viz_color(COL_RESET);
			} else if (ch == '+' && (r == 0 || r == GRID_H - 1 || c == 0 || c == GRID_W - 1)) {
				viz_color(COL_DIM);
				putchar(ch);
				viz_color(COL_RESET);
			} else if (ch == '+') {
				viz_color(COL_CYAN);
				putchar(ch);
				viz_color(COL_RESET);
			} else if (ch == '~') {
				viz_color(COL_DIM);
				putchar(ch);
				viz_color(COL_RESET);
			} else if (ch == '|' || ch == '-') {
				viz_color(COL_DIM);
				putchar(ch);
				viz_color(COL_RESET);
			} else {
				putchar(ch);
			}
		}
		putchar('\n');
	}

	xmid = (g->xmin + g->xmax) / 2.0;
	viz_color(COL_DIM);
	printf("       %-*.*f", GRID_W / 2, 2, g->xmin);
	printf("%*.2f\n", GRID_W - GRID_W / 2, g->xmax);
	printf("       %*s%.2f\n", GRID_W / 2 - 2, "", xmid);
	viz_color(COL_RESET);
}

int viz_grid_map_x(Grid *g, float x) {
	int cx = 1 + (int)((x - g->xmin) / (g->xmax - g->xmin) * (GRID_W - 3));
	if (cx < 1) cx = 1;
	if (cx > GRID_W - 2) cx = GRID_W - 2;
	return cx;
}

int viz_grid_map_y(Grid *g, float y) {
	int cy = (GRID_H - 2) - (int)((y - g->ymin) / (g->ymax - g->ymin) * (GRID_H - 3));
	if (cy < 1) cy = 1;
	if (cy > GRID_H - 2) cy = GRID_H - 2;
	return cy;
}

void viz_grid_point(Grid *g, float x, float y, char ch) {
	int cx = viz_grid_map_x(g, x);
	int cy = viz_grid_map_y(g, y);
	g->cells[cy][cx] = ch;
}

void viz_grid_trajectory(Grid *g, Matrix xs, Matrix ys, char ch) {
	int i, n;
	n = xs->width > ys->width ? ys->width : xs->width;
	for (i = 0; i < n; i++) {
		float x = elem(xs, 0, i);
		float y = elem(ys, 0, i);
		viz_grid_point(g, x, y, ch);
	}
}

void viz_grid_ellipse(Grid *g, float cx, float cy, Matrix cov, char ch) {
	int i, npts = 36;
	float a, b, theta, t;
	Matrix A, Vec, Val;

	/* extract 2x2 position covariance and decompose */
	A = newMatrix(2, 2);
	setElem(A, 0, 0, elem(cov, 0, 0));
	setElem(A, 0, 1, elem(cov, 0, 1));
	setElem(A, 1, 0, elem(cov, 1, 0));
	setElem(A, 1, 1, elem(cov, 1, 1));

	Vec = newMatrix(2, 2);
	Val = newMatrix(2, 2);
	eig(&A, &Vec, &Val);

	/* semi-axis lengths: k * sqrt(eigenvalue), k=2.0 for ~95% */
	a = 2.0 * sqrt(elem(Val, 0, 0));
	b = 2.0 * sqrt(elem(Val, 1, 1));

	/* rotation angle from eigenvectors */
	theta = atan2(elem(Vec, 1, 0), elem(Vec, 0, 0));

	/* Bug 5: no terminal aspect ratio compensation
	 * chars are ~2x taller than wide, should multiply x by 2
	 * but we don't — ellipse will look squashed horizontally */

	/* plot parametric ellipse */
	for (i = 0; i < npts; i++) {
		float ex, ey;
		t = 2.0 * pi * i / npts;
		ex = cx + a * cos(t) * cos(theta) - b * sin(t) * sin(theta);
		ey = cy + a * cos(t) * sin(theta) + b * sin(t) * cos(theta);

		/* only plot if inside grid bounds (with some margin) */
		if (ex >= g->xmin && ex <= g->xmax && ey >= g->ymin && ey <= g->ymax) {
			int gx = viz_grid_map_x(g, ex);
			int gy = viz_grid_map_y(g, ey);
			/* don't overwrite existing markers */
			if (g->cells[gy][gx] == ' ')
				g->cells[gy][gx] = ch;
		}
	}

	freeMatrix(A);
	freeMatrix(Vec);
	freeMatrix(Val);
}

void viz_convergence_bar(float trace_p, float trace_p0, int width) {
	int i, filled;
	float ratio;

	if (trace_p0 <= 0) trace_p0 = 1.0;
	ratio = 1.0 - trace_p / trace_p0;

	filled = (int)(ratio * width);

	printf("[");
	viz_color(COL_GREEN);
	for (i = 0; i < filled; i++) printf("=");
	viz_color(COL_RESET);
	for (i = filled; i < width; i++) printf(" ");
	printf("]");
}
