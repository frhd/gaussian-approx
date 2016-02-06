#ifndef sim_h
#define sim_h

#include "matrix.h"

/* trajectory types */
#define SIM_CIRCLE  0
#define SIM_LINE    1
#define SIM_FIGURE8 2
#define SIM_RANDOM  3

typedef struct {
	Matrix true_pos;     /* nsteps x 2 */
	Matrix true_vel;     /* nsteps x 2 */
	Matrix measurements; /* nsteps x 2 (with noise) */
	int nsteps;
	float dt;
} Scenario;

/* trajectory generators — return nsteps x 2 matrices of [x,y] positions */
Matrix sim_trajectory_line(int nsteps, float dt, float vx, float vy);
Matrix sim_trajectory_circle(int nsteps, float dt, float radius);
Matrix sim_trajectory_figure8(int nsteps, float dt, float amplitude);

/* generate noisy measurements from true positions */
Matrix sim_measurements(Matrix true_pos, float noise_std);

/* scenario factory */
Scenario *sim_create_scenario(int type, int nsteps, float dt, float noise);
void sim_free_scenario(Scenario *s);

#endif
