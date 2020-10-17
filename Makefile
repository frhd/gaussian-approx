# Makefile for vizga — ASCII Kalman filter visualizer
#
# Usage:
#   make        — build vizga binary
#   make run    — build and run default demo
#   make demo   — run quick demo (circle, 80 steps)
#   make clean  — remove build artifacts
#   make help   — show this information

CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

OBJS = matrix.o eig.o gaussianApprox.o gaussianEstimator.o viz.o sim.o export.o main.o

vizga: $(OBJS)
	$(CC) -o vizga $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o vizga

run: vizga
	./vizga

demo: vizga
	./vizga -t circle -n 80 --speed 80

help:
	@echo "vizga — ASCII Kalman filter visualizer"
	@echo ""
	@echo "Targets:"
	@echo "  make        — build vizga binary"
	@echo "  make run    — build and run default demo"
	@echo "  make demo   — run quick demo (circle, 80 steps)"
	@echo "  make clean  — remove build artifacts"
	@echo "  make help   — show this information"
	@echo ""
	@echo "Usage: ./vizga [options]"
	@echo "  -m <mode>   2d, 1d, multi, rot, compare, test, grid"
	@echo "  -t <traj>   circle, line, fig8, random"
	@echo "  -n <steps>  number of steps"
	@echo "  --speed N   animation delay in ms"
	@echo "  -h          full help"

.PHONY: clean run demo help
