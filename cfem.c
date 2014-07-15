#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef WITH_AP
/*
 * It is pretty sloppy to include the whole library here, but we use most of it
 * and this makes linking a whole lot simpler (no multiple so files)
 */
#define USE_ROW_MAJOR
#define LAPACKINDEX int
#include "argyris_pack.c"
#include "argyris_pack.h"
#endif

#include "cfem.h"
#include "sample.c"

#include "misc.c"
#include "test_functions.c"
#include "local_matrices.c"
#include "mesh.c"

#include "sparse_triplet.c"
#include "loadvector.c"

#include "global_matrices.c"

#ifdef WITH_AP
#include "cfem_ap.c"
#endif
