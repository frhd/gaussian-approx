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

Matrix sim_trajectory_circle(int nsteps, float dt, float radius) {
	int i;
	float omega = 0.5;	/* angular velocity */
	Matrix pos = newMatrix(nsteps, 2);

	for (i = 0; i < nsteps; i++) {
		float t = i * dt;
		float x = radius * cos(omega * t);
		float y = radius * sin(omega * t);
		/* add small process noise */
		x += 0.05 * randn();
		y += 0.05 * randn();
		setElem(pos, i, 0, x);
		setElem(pos, i, 1, y);
	}
	return pos;
}

void sim_free_scenario(Scenario *s) {
	if (!s) return;
	freeMatrix(s->true_pos);
	freeMatrix(s->true_vel);
	freeMatrix(s->measurements);
	free(s);
}

Matrix sim_measurements(Matrix true_pos, float noise_std) {
	int i, nsteps;
	Matrix meas;

	nsteps = true_pos->height;
	meas = newMatrix(nsteps, 2);

	for (i = 0; i < nsteps; i++) {
		setElem(meas, i, 0, elem(true_pos, i, 0) + noise_std * randn());
		setElem(meas, i, 1, elem(true_pos, i, 1) + noise_std * randn());
	}
	return meas;
}
