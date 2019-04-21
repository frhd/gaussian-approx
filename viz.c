#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include "viz.h"
#include "eig.h"
#include "tracker.h"

static struct termios orig_termios;
static int raw_mode_active = 0;

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

/* color a grid cell based on its character */
static void viz_grid_putchar(char ch, int r, int c) {
	if (ch == '.') {
		viz_color(COL_GREEN);
		putchar(ch);
		viz_color(COL_RESET);
	} else if (ch == ',') {
		viz_color(COL_DIM);
		putchar('.');
		viz_color(COL_RESET);
	} else if (ch == 'o') {
		viz_color(COL_YELLOW);
		putchar(ch);
		viz_color(COL_RESET);
	} else if (ch == 'O') {
		viz_color(COL_BOLD);
		viz_color(COL_YELLOW);
		putchar('o');
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
	} else if (ch == 'A' || ch == 'a') {
		viz_color(ch == 'A' ? COL_BOLD : "");
		viz_color(COL_GREEN);
		putchar(ch);
		viz_color(COL_RESET);
	} else if (ch == 'B' || ch == 'b') {
		viz_color(ch == 'B' ? COL_BOLD : "");
		viz_color(COL_YELLOW);
		putchar(ch);
		viz_color(COL_RESET);
	} else if (ch == 'C' || ch == 'c') {
		viz_color(ch == 'C' ? COL_BOLD : "");
		viz_color(COL_CYAN);
		putchar(ch);
		viz_color(COL_RESET);
	} else if (ch == 'D' || ch == 'd') {
		viz_color(ch == 'D' ? COL_BOLD : "");
		viz_color(COL_RED);
		putchar(ch);
		viz_color(COL_RESET);
	} else if (ch == '?') {
		viz_color(COL_DIM);
		putchar(ch);
		viz_color(COL_RESET);
	} else {
		putchar(ch);
	}
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
			viz_grid_putchar(ch, r, c);
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

void viz_grid_print_at(Grid *g, int row, int col) {
	int r, c;
	char label[16];
	float yval, xmid;

	for (r = 0; r < GRID_H; r++) {
		viz_cursor_move(row + r, col);
		yval = g->ymax - (g->ymax - g->ymin) * r / (GRID_H - 1);
		if (r == 0 || r == GRID_H / 2 || r == GRID_H - 1) {
			snprintf(label, sizeof(label), "%6.2f ", yval);
			printf("%s", label);
		} else {
			printf("       ");
		}
		for (c = 0; c < GRID_W; c++) {
			viz_grid_putchar(g->cells[r][c], r, c);
		}
	}

	/* x axis labels */
	xmid = (g->xmin + g->xmax) / 2.0;
	viz_cursor_move(row + GRID_H, col);
	viz_color(COL_DIM);
	printf("       %-*.*f", GRID_W / 2, 2, g->xmin);
	printf("%*.2f", GRID_W - GRID_W / 2, g->xmax);
	viz_cursor_move(row + GRID_H + 1, col);
	printf("       %*s%.2f", GRID_W / 2 - 2, "", xmid);
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

	/* extract 2x2 covariance and decompose */
	A = newMatrix(2, 2);
	setElem(A, 0, 0, elem(cov, 0, 0));
	setElem(A, 0, 1, elem(cov, 0, 1));
	setElem(A, 1, 0, elem(cov, 1, 0));
	setElem(A, 1, 1, elem(cov, 1, 1));

	Vec = newMatrix(2, 2);
	Val = newMatrix(2, 2);
	eig(&A, &Vec, &Val);

	/* check for singular/degenerate covariance */
	if (elem(Val, 0, 0) <= 0 || elem(Val, 1, 1) <= 0) {
		freeMatrix(A);
		freeMatrix(Vec);
		freeMatrix(Val);
		return;
	}

	/* semi-axis lengths: k * sqrt(eigenvalue), k=2.0 for ~95% confidence */
	a = 2.0 * sqrt(elem(Val, 0, 0));
	b = 2.0 * sqrt(elem(Val, 1, 1));

	/* rotation angle from eigenvectors */
	theta = atan2(elem(Vec, 1, 0), elem(Vec, 0, 0));

	/* plot parametric ellipse */
	for (i = 0; i < npts; i++) {
		float ex, ey, px, py;
		t = 2.0 * pi * i / npts;
		px = a * cos(t) * cos(theta) - b * sin(t) * sin(theta);
		py = a * cos(t) * sin(theta) + b * sin(t) * cos(theta);
		/* scale x by aspect ratio to compensate for char height:width ~2:1 */
		ex = cx + px * 2.0;
		ey = cy + py;

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
	if (ratio < 0) ratio = 0;
	if (ratio > 1.0) ratio = 1.0;

	filled = (int)(ratio * width);

	printf("[");
	viz_color(COL_GREEN);
	for (i = 0; i < filled; i++) printf("=");
	if (filled < width) printf(">");
	viz_color(COL_RESET);
	for (i = filled + 1; i < width; i++) printf(" ");
	printf("]");
}

void viz_progress_bar(int step, int total, int width) {
	int i, filled;
	if (total <= 0) total = 1;
	filled = (int)((float)(step + 1) / total * width);
	if (filled > width) filled = width;

	printf("[");
	viz_color(COL_CYAN);
	for (i = 0; i < filled; i++) printf("#");
	viz_color(COL_RESET);
	viz_color(COL_DIM);
	for (i = filled; i < width; i++) printf(".");
	viz_color(COL_RESET);
	printf("] %d/%d", step + 1, total);
}

void viz_cursor_move(int row, int col) {
	printf("\033[%d;%dH", row, col);
}

void viz_clear_screen(void) {
	printf("\033[2J\033[H");
}

int viz_term_width(void) {
	struct winsize ws;
	if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
		return ws.ws_col;
	return 80;
}

void viz_panel_border(Panel *p) {
	int r, c;
	/* top border */
	viz_cursor_move(p->row, p->col);
	viz_color(COL_DIM);
	putchar('+');
	for (c = 1; c < p->width - 1; c++) putchar('-');
	putchar('+');

	/* bottom border */
	viz_cursor_move(p->row + p->height - 1, p->col);
	putchar('+');
	for (c = 1; c < p->width - 1; c++) putchar('-');
	putchar('+');

	/* side borders */
	for (r = 1; r < p->height - 1; r++) {
		viz_cursor_move(p->row + r, p->col);
		putchar('|');
		viz_cursor_move(p->row + r, p->col + p->width - 1);
		putchar('|');
	}
	viz_color(COL_RESET);
}

void viz_panel_text(Panel *p, int line, const char *text) {
	int maxw = p->width - 2;
	if (line < 0 || line >= p->height - 2) return;
	viz_cursor_move(p->row + 1 + line, p->col + 1);
	printf("%-*.*s", maxw, maxw, text);
}

/* Rodrigues rotation: R = I + sin(t)/t * K + (1-cos(t))/t^2 * K^2
 * where K is skew-symmetric of rotation vector, t = |rot| */
static void rodrigues(float rot[3], float R[3][3]) {
	float theta = sqrt(rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2]);
	float K[3][3] = {{0, -rot[2], rot[1]},
	                  {rot[2], 0, -rot[0]},
	                  {-rot[1], rot[0], 0}};
	int i, j, k;

	if (theta < 1e-6) {
		/* identity */
		for (i = 0; i < 3; i++)
			for (j = 0; j < 3; j++)
				R[i][j] = (i == j) ? 1.0 : 0.0;
		return;
	}

	{
		float s = sin(theta) / theta;
		float c = (1.0 - cos(theta)) / (theta * theta);
		float K2[3][3];

		for (i = 0; i < 3; i++)
			for (j = 0; j < 3; j++) {
				K2[i][j] = 0;
				for (k = 0; k < 3; k++)
					K2[i][j] += K[i][k] * K[k][j];
			}

		for (i = 0; i < 3; i++)
			for (j = 0; j < 3; j++)
				R[i][j] = (i == j ? 1.0 : 0.0) + s * K[i][j] + c * K2[i][j];
	}
}

/* simple orthographic 3d -> 2d projection
 * isometric-ish: px = x + 0.5*z, py = y + 0.3*z */
void viz_project_3d(float x, float y, float z, float *px, float *py) {
	*px = x + 0.5 * z;
	*py = y + 0.3 * z;
}

/* draw rotated 3d axes on grid using rotation vector */
void viz_grid_axes_3d(Grid *g, float cx, float cy, float rot[3]) {
	float R[3][3];
	float axes[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
	char labels[] = {'x', 'y', 'z'};
	float len = 2.0;
	int a, i;

	rodrigues(rot, R);

	for (a = 0; a < 3; a++) {
		float rx = 0, ry = 0, rz = 0;
		for (i = 0; i < 3; i++) {
			rx += R[0][i] * axes[a][i];
			ry += R[1][i] * axes[a][i];
			rz += R[2][i] * axes[a][i];
		}
		rx *= len; ry *= len; rz *= len;

		{
			int npts = 12;
			int k;
			for (k = 0; k <= npts; k++) {
				float t = (float)k / npts;
				float px, py;
				viz_project_3d(rx * t, ry * t, rz * t, &px, &py);
				px = cx + px * 2.0;
				py = cy + py;
				if (px >= g->xmin && px <= g->xmax &&
				    py >= g->ymin && py <= g->ymax) {
					int gx = viz_grid_map_x(g, px);
					int gy = viz_grid_map_y(g, py);
					char ch = (k == npts) ? labels[a] : '-';
					if (g->cells[gy][gx] == ' ' || k == npts)
						g->cells[gy][gx] = ch;
				}
			}
		}
	}
}

void term_restore(void) {
	if (raw_mode_active) {
		tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
		raw_mode_active = 0;
	}
}

void term_raw_mode(void) {
	struct termios raw;

	if (!isatty(STDIN_FILENO)) return;

	tcgetattr(STDIN_FILENO, &orig_termios);
	raw = orig_termios;
	raw.c_lflag &= ~(ICANON | ECHO);
	raw.c_cc[VMIN] = 0;
	raw.c_cc[VTIME] = 0;
	tcsetattr(STDIN_FILENO, TCSANOW, &raw);
	raw_mode_active = 1;
	atexit(term_restore);	/* safe to call multiple times */
}

int term_kbhit(void) {
	int n = 0;
	ioctl(STDIN_FILENO, FIONREAD, &n);
	return n > 0;
}

int term_getchar(void) {
	unsigned char c;
	if (read(STDIN_FILENO, &c, 1) == 1)
		return c;
	return -1;
}
