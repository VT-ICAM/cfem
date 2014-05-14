#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cfem.h"
#include "sparse_triplet.c"
#include "local_matrices.c"

int main()
{
        int i, status;
        int num_points = 3;
        int num_nodes = 6;
        int num_elements = 4;
        int num_basis_functions = 3;
        int length = num_elements*ipow(num_basis_functions, 2);
        double nodes[] = {0.0, 1.0,
                          1.0, 1.0,
                          2.0, 1.0,
                          0.0, 0.0,
                          1.0, 0.0,
                          2.0, 0.0};
        int elements[] = {0, 1, 4,
                          0, 3, 4,
                          1, 2, 5,
                          1, 4, 5};
        cf_mesh_s mesh = {.num_nodes = num_nodes, .nodes = nodes,
                          .num_elements = num_elements,
                          .num_basis_functions = num_basis_functions,
                          .elements = elements};

        double weights[] = {1.0, 1.0, 1.0};
        double ref_values[] = {1.0, 1.0, 1.0};
        const cf_ref_arrays_s ref_arrays = {.num_points = num_points,
                                            .num_basis_functions = num_basis_functions,
                                            .weights = weights,
                                            .values = ref_values,
                                            .dx = ref_values,
                                            .dy = ref_values,
                                            .dxx = ref_values,
                                            .dxy = ref_values,
                                            .dyy = ref_values};


        int *rows = (int*) malloc(length*sizeof(int));
        int *columns = (int*) malloc(length*sizeof(int));
        double *values = (double*) malloc(length*sizeof(double));

        cf_triplet_matrix_s triplets = {.length = length, .rows = rows,
                                        .columns = columns, .values = values};

        printf("calling cf_build_triplet_matrix...");
        status = cf_build_triplet_matrix(&cf_local_stiffness, &mesh, &ref_arrays, &triplets);
        printf(" done. status is %d\n", status);
        for (i = 0; i < length; i++) {
                printf("%d %d %f\n", rows[i], columns[i], values[i]);
        }


        free(rows);
        free(columns);
        free(values);

        return 0;
}
