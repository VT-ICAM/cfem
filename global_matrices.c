/** @file */
/**
 * @brief Build a mass matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_build_mass(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                  cf_convection_f convection_function,
                  cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_local_mass, &mesh, &ref_data,
                                         convection_function, &matrix);
        return status;
}

/**
 * @brief Build a convection matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_build_convection(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                        cf_convection_f convection_function,
                        cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_local_convection, &mesh, &ref_data,
                                         convection_function, &matrix);
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
int cf_build_stiffness(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                       cf_convection_f convection_function,
                       cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_local_stiffness, &mesh, &ref_data,
                                         convection_function, &matrix);
        return status;
}

/**
 * @brief Build a hessian matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_build_hessian(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                     cf_convection_f convection_function,
                     cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_local_hessian, &mesh, &ref_data,
                                         convection_function, &matrix);
        return status;
}
