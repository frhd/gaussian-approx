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

.PHONY: clean run
