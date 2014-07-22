CFLAGS = -std=c99 -Wall -pedantic -Wextra -Wno-unused-parameter -O3
LDFLAGS=-L../../ArgyrisPack/

a.out: *.c
	$(CC) $(CFLAGS) main.c -lblas

libcfem.so: *.c
	$(CC) $(CFLAGS) -fPIC cfem.c -lblas -shared -o libcfem.so

autogenerated_functions: funcs.c
	$(CC) $(CFLAGS) -fPIC funcs.c -shared -o libfuncs.so

withap: *.c
	$(CC) $(CFLAGS) -I../../ArgyrisPack/ap/numeric -fPIC -DWITH_AP cfem.c      \
    -shared -o libcfem.so -lblas
