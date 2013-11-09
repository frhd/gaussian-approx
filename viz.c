#include <stdio.h>
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
		for (i = 0; i < center - n; i++) printf(" ");
		for (i = 0; i < n; i++) printf("#");
		printf("\n");
	}
}

void viz_vector(Matrix v) {
	int i;
	float max_val = 0;

	for (i = 0; i < v->height; i++) {
		float val = fabs(elem(v, i, 0));
		if (val > max_val) max_val = val;
	}
	if (max_val == 0) max_val = 1.0;

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

	for (x = 0; x < width; x++) printf("-");
	printf("\n");

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

	for (x = 0; x < width; x++) {
		float xv = xmin + x * dx;
		int marked = 0;

		if (fabs(xv - mean) < dx * 0.5) {
			printf("+");
			marked = 1;
		}

		if (!marked) {
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

	printf("(%d sigma points)\n", npts - 1);
}
