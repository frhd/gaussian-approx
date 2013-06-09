CC = gcc
CFLAGS = -Wall
LDFLAGS = -lm

OBJS = matrix.o eig.o gaussianApprox.o gaussianEstimator.o

all: $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
