#ifndef export_h
#define export_h

#include <stdio.h>

FILE *export_open(const char *filename);
void export_header_2d(FILE *f);
void export_row_2d(FILE *f, int step, float t,
	float true_x, float true_y, float meas_x, float meas_y,
	float est_x, float est_y, float est_vx, float est_vy,
	float cov_xx, float cov_yy, float rmse);
void export_close(FILE *f);

#endif
