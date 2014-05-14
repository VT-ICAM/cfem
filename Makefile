CC = gcc
CFLAGS = -std=c99 -Wall -pedantic -Wextra -O3

a.out: *.c
	$(CC) $(CFLAGS) main.c -lblas

libcfem.so: *.c
	$(CC) $(CFLAGS) -fPIC global_matrices.c -lblas -shared -o libcfem.so
