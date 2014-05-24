#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "viz.h"
#include "tracker.h"

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
			if (pdf[x] >= ythresh)
				printf("*");
			else
				printf(" ");
		}
		printf("\n");
	}

	/* x axis */
	for (x = 0; x < width; x++) printf("-");
	printf("\n");

	/* labels */
	printf("%-*.*f", width / 2, 2, xmin);
	printf("%*.2f", width - width / 2, xmax);
	printf("\n");
	printf("%*s%.2f\n", width / 2 - 2, "", mean);

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
			printf("+");
			marked = 1;
		}

		if (!marked) {
			/* check sigma points */
			for (i = 0; i < npts; i++) {
				float sp = mean + elem(m_opt, 0, i) * sigma;
				if (fabs(xv - sp) < dx * 0.5) {
					printf("|");
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

	/* fill with spaces */
	for (r = 0; r < GRID_H; r++)
		for (c = 0; c < GRID_W; c++)
			g->cells[r][c] = ' ';

	/* draw border */
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
	for (r = 0; r < GRID_H; r++) {
		for (c = 0; c < GRID_W; c++)
			putchar(g->cells[r][c]);
		putchar('\n');
	}
}

int viz_grid_map_x(Grid *g, float x) {
	int cx = 1 + (int)((x - g->xmin) / (g->xmax - g->xmin) * (GRID_W - 4));
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
