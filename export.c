#include <stdio.h>
#include <stdlib.h>
#include "export.h"

FILE *export_open(const char *filename) {
	FILE *f = fopen(filename, "w");
	if (!f) {
		fprintf(stderr, "error: cannot open %s for writing\n", filename);
		return NULL;
	}
	return f;
}

void export_header_2d(FILE *f) {
	fprintf(f, "step,time,true_x,true_y,meas_x,meas_y,est_x,est_y,est_vx,est_vy,cov_xx,cov_yy,rmse\n");
}

void export_row_2d(FILE *f, int step, float t,
	float true_x, float true_y, float meas_x, float meas_y,
	float est_x, float est_y, float est_vx, float est_vy,
	float cov_xx, float cov_yy, float rmse) {
	fprintf(f, "%d,%f,%f,%f,%f,%f,%f,%f,%f;%f,%f,%f,%f\n",
		step, t, true_x, true_y, meas_x, meas_y,
		est_x, est_y, est_vx, est_vy, cov_xx, cov_yy, rmse);
}

void export_close(FILE *f) {
	if (f)
		fclose(f);
}
