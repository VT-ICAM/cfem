/** @file */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cfem.h"
#include "sparse_triplet.c"
#include "local_matrices.c"

/**
 * @brief Build a mass matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_build_mass(const mesh_s mesh, const ref_arrays_s ref_data,
                  triplet_matrix_s matrix) {
        int status;
        status = cf_build_triplet_matrix(&cf_local_mass, &mesh, &ref_data, &matrix);
        return status;
}

/**
 * @brief Build a stiffness matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_build_stiffness(const mesh_s mesh, const ref_arrays_s ref_data,
                       triplet_matrix_s matrix) {
        int status;
        status = cf_build_triplet_matrix(&cf_local_stiffness, &mesh, &ref_data, &matrix);
        return status;
}
