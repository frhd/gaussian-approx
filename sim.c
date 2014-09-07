#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sim.h"
#include "noise.h"
#include "tracker.h"

/* Box-Muller transform */
float randn(void) {
	float u1, u2;
	do {
		u1 = (float)rand() / RAND_MAX;
	} while (u1 == 0);
	u2 = (float)rand() / RAND_MAX;
	return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2);
}

Matrix sim_trajectory_line(int nsteps, float dt, float vx, float vy) {
	int i;
	float x = 0, y = 0;
	Matrix pos = newMatrix(nsteps, 2);

	for (i = 0; i < nsteps; i++) {
		x += vx * dt + 0.1 * randn();
		y += vy * dt + 0.1 * randn();
		setElem(pos, i, 0, x);
		setElem(pos, i, 1, y);
	}
	return pos;
}
