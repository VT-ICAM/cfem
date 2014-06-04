CC = gcc
CFLAGS = -std=c99 -Wall -pedantic -Wextra -O0 -ggdb

a.out: *.c
	$(CC) $(CFLAGS) main.c -lblas

libcfem.so: *.c
	$(CC) $(CFLAGS) -fPIC cfem.c -lblas -shared -o libcfem.so
