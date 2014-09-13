#ifndef sim_h
#define sim_h

#include "matrix.h"

/* trajectory generators — return nsteps x 2 matrices of [x,y] positions */
Matrix sim_trajectory_line(int nsteps, float dt, float vx, float vy);
Matrix sim_trajectory_circle(int nsteps, float dt, float radius);

/* generate noisy measurements from true positions */
Matrix sim_measurements(Matrix true_pos, float noise_std);

#endif
