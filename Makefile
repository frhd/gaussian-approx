CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm

OBJS = matrix.o eig.o gaussianApprox.o gaussianEstimator.o main.o

vizga: $(OBJS)
	$(CC) -o vizga $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o vizga

run: vizga
	./vizga
