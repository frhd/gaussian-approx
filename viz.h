#ifndef viz_h
#define viz_h

#include "matrix.h"

#define GRID_W 60
#define GRID_H 30

typedef struct {
	char cells[GRID_H][GRID_W];
	float xmin, xmax, ymin, ymax;
} Grid;

void viz_bar(float value, float max_val, int width);
void viz_vector(Matrix v);
void viz_gaussian_1d(float mean, float sigma, int width, int height);
void viz_sigma_points_1d(float mean, float sigma, Matrix m_opt, int width);

void viz_grid_init(Grid *g, float xmin, float xmax, float ymin, float ymax);
void viz_grid_print(Grid *g);
int  viz_grid_map_x(Grid *g, float x);
int  viz_grid_map_y(Grid *g, float y);

#endif
