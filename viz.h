#ifndef viz_h
#define viz_h

#include "matrix.h"

#define GRID_W 60
#define GRID_H 30

/* ansi color codes */
#define COL_RESET   "\033[0m"
#define COL_RED     "\033[31m"
#define COL_GREEN   "\033[32m"
#define COL_YELLOW  "\033[33m"
#define COL_BLUE    "\033[34m"
#define COL_CYAN    "\033[36m"
#define COL_DIM     "\033[2m"
#define COL_BOLD    "\033[1m"

extern int viz_color_enabled;

typedef struct {
	char cells[GRID_H][GRID_W];
	float xmin, xmax, ymin, ymax;
} Grid;

/* panel layout */
typedef struct {
	int row, col;
	int width, height;
} Panel;

/* 1d visualization */
void viz_bar(float value, float max_val, int width);
void viz_vector(Matrix v);
void viz_gaussian_1d(float mean, float sigma, int width, int height);
void viz_sigma_points_1d(float mean, float sigma, Matrix m_opt, int width);
void viz_sigma_weights(Matrix m_opt);
void viz_sigma_points_2d(Grid *g, Matrix xEst, Matrix CEst, Matrix m_opt, int post_transform);

/* 2d grid */
void viz_grid_init(Grid *g, float xmin, float xmax, float ymin, float ymax);
void viz_grid_print(Grid *g);
void viz_grid_print_at(Grid *g, int row, int col);
int  viz_grid_map_x(Grid *g, float x);
int  viz_grid_map_y(Grid *g, float y);
void viz_grid_point(Grid *g, float x, float y, char ch);
void viz_grid_trajectory(Grid *g, Matrix xs, Matrix ys, char ch);
void viz_grid_ellipse(Grid *g, float cx, float cy, Matrix cov, char ch);

/* color and display helpers */
void viz_color(const char *code);
void viz_convergence_bar(float trace_p, float trace_p0, int width);
void viz_progress_bar(int step, int total, int width);
void viz_sparkline(float *values, int count, int width);

/* panel layout */
void viz_cursor_move(int row, int col);
void viz_panel_border(Panel *p);
void viz_panel_text(Panel *p, int line, const char *text);
void viz_clear_screen(void);
int  viz_term_width(void);

/* 3d projection */
void viz_project_3d(float x, float y, float z, float *px, float *py);
void viz_grid_axes_3d(Grid *g, float cx, float cy, float rot[3]);

/* terminal raw mode for interactive input */
void term_raw_mode(void);
void term_restore(void);
int  term_kbhit(void);
int  term_getchar(void);

#endif
