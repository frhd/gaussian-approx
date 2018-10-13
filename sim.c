#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sim.h"
#include "noise.h"
#include "tracker.h"

/* Box-Muller transform — generates N(0,1) samples.
 * uses rejection for u1=0 to avoid log(0) */
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
		x += 0.05 * randn();
		y += 0.05 * randn();
		setElem(pos, i, 0, x);
		setElem(pos, i, 1, y);
	}
	return pos;
}

Matrix sim_trajectory_figure8(int nsteps, float dt, float amplitude) {
	int i;
	float omega = 2.0 * pi / (100.0 * dt);
	Matrix pos = newMatrix(nsteps, 2);

	for (i = 0; i < nsteps; i++) {
		float t = i * dt;
		float x = amplitude * sin(omega * t);
		float y = amplitude * sin(2.0 * omega * t) / 2.0;
		x += 0.05 * randn();
		y += 0.05 * randn();
		setElem(pos, i, 0, x);
		setElem(pos, i, 1, y);
	}
	return pos;
}

Matrix sim_trajectory_random_walk(int nsteps, float dt) {
	int i;
	float x = 0, y = 0;
	float vx = 1.0, vy = 0.5;
	Matrix pos = newMatrix(nsteps, 2);

	for (i = 0; i < nsteps; i++) {
		/* randomly perturb velocity */
		vx += 0.3 * randn();
		vy += 0.3 * randn();

		x += vx * dt;
		y += vy * dt;
		setElem(pos, i, 0, x);
		setElem(pos, i, 1, y);
	}
	return pos;
}

/* compute velocity from position via finite differences.
 * first step uses forward diff, rest use backward */
static Matrix compute_velocities(Matrix pos, float dt) {
	int i, nsteps;
	Matrix vel;

	nsteps = pos->height;
	vel = newMatrix(nsteps, 2);

	/* first step: use forward difference */
	if (nsteps > 1) {
		setElem(vel, 0, 0, (elem(pos, 1, 0) - elem(pos, 0, 0)) / dt);
		setElem(vel, 0, 1, (elem(pos, 1, 1) - elem(pos, 0, 1)) / dt);
	}

	for (i = 1; i < nsteps; i++) {
		setElem(vel, i, 0, (elem(pos, i, 0) - elem(pos, i - 1, 0)) / dt);
		setElem(vel, i, 1, (elem(pos, i, 1) - elem(pos, i - 1, 1)) / dt);
	}

	return vel;
}

Scenario *sim_create_scenario(int type, int nsteps, float dt, float noise) {
	Scenario *s = (Scenario *)malloc(sizeof(Scenario));
	s->nsteps = nsteps;
	s->dt = dt;

	switch (type) {
	case SIM_LINE:
		s->true_pos = sim_trajectory_line(nsteps, dt, 3.0, 1.5);
		break;
	case SIM_CIRCLE:
		s->true_pos = sim_trajectory_circle(nsteps, dt, 5.0);
		break;
	case SIM_FIGURE8:
		s->true_pos = sim_trajectory_figure8(nsteps, dt, 5.0);
		break;
	case SIM_RANDOM:
		s->true_pos = sim_trajectory_random_walk(nsteps, dt);
		break;
	default:
		fprintf(stderr, "unknown trajectory type %d\n", type);
		s->true_pos = sim_trajectory_circle(nsteps, dt, 5.0);
		break;
	}

	s->true_vel = compute_velocities(s->true_pos, dt);
	s->measurements = sim_measurements(s->true_pos, noise);
	return s;
}

const char *sim_trajectory_name(int type) {
	switch (type) {
	case SIM_LINE:    return "line";
	case SIM_CIRCLE:  return "circle";
	case SIM_FIGURE8: return "figure-8";
	case SIM_RANDOM:  return "random walk";
	default:          return "unknown";
	}
}

void sim_free_scenario(Scenario *s) {
	if (!s) return;
	if (s->true_pos) freeMatrix(s->true_pos);
	if (s->true_vel) freeMatrix(s->true_vel);
	if (s->measurements) freeMatrix(s->measurements);
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

int sim_measurement_dropout(void) {
	return ((float)rand() / RAND_MAX) < 0.1;
}

void sim_multi_scenario(Target *targets, int ntargets, int nsteps, float dt, float noise) {
	int k;
	/* trajectory types to assign: cycle through available types */
	int types[] = {SIM_CIRCLE, SIM_LINE, SIM_FIGURE8, SIM_RANDOM};
	char markers[] = {'A', 'B', 'C', 'D'};
	/* position offsets so targets don't overlap */
	float offsets[][2] = {{0, 0}, {8, 4}, {-6, 6}, {4, -8}};

	for (k = 0; k < ntargets && k < MAX_TARGETS; k++) {
		int ttype = types[k % 4];
		targets[k].scen = sim_create_scenario(ttype, nsteps, dt, noise);
		targets[k].marker = markers[k];
		targets[k].active = 1;
		targets[k].xEst = NULL;
		targets[k].CEst = NULL;

		/* apply position offset to separate targets */
		if (targets[k].scen && k > 0) {
			int i;
			Matrix pos = targets[k].scen->true_pos;
			Matrix meas = targets[k].scen->measurements;
			for (i = 0; i < nsteps; i++) {
				setElem(pos, i, 0, elem(pos, i, 0) + offsets[k][0]);
				setElem(pos, i, 1, elem(pos, i, 1) + offsets[k][1]);
				setElem(meas, i, 0, elem(meas, i, 0) + offsets[k][0]);
				setElem(meas, i, 1, elem(meas, i, 1) + offsets[k][1]);
			}
		}
	}
}

void sim_free_targets(Target *targets, int ntargets) {
	int k;
	for (k = 0; k < ntargets; k++) {
		if (targets[k].scen)
			sim_free_scenario(targets[k].scen);
		if (targets[k].xEst)
			freeMatrix(targets[k].xEst);
		if (targets[k].CEst)
			freeMatrix(targets[k].CEst);
	}
}
