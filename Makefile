CC = gcc
CFLAGS = -Wall
LDFLAGS = -lm

OBJS = matrix.o eig.o gaussianApprox.o gaussianEstimator.o main.o

vizga: $(OBJS)
	$(CC) -o vizga $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: vizga
	./vizga

clean:
	rm -f *.o vizga
