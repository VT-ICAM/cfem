/** @file */
void cf_ap_physical_maps(const cf_local_element_s* local_element, double C[static 441],
                         double B[static 4], double b[static 2])
{
        ap_physical_maps((double*) local_element->xs, (double*) local_element->ys,
                         C, B, b);
}

/**
 * @brief Calculate the local mass matrix.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field. No SUPG stabilization will be done if this is NULL.
 * @param[out] mass_matrix Pointer to the array which will be filled in with
 * mass matrix data. The array must be of length (at least) 441.
 */
int cf_ap_local_mass(const cf_ref_arrays_s* ref_data,
                     const cf_local_element_s* local_element,
                     cf_convection_f convection_function,
                     double mass_matrix[static 441])
{
        int i;
        int num_points = ref_data->num_points;
        double function_values[21*num_points];
        double function_values_scaled[21*num_points];
        double weights_scaled[num_points];
        double C[441];
        /* These are not used, but the original ArgyrisPack code expects to
         * compute them.
         */
        double B[4];
        double b[2];
        int status = 0;
        /* stuff for DGEMM. */
        int i_twentyone = 21;

        cf_ap_physical_maps(local_element, C, B, b);
        ap_physical_values(C, ref_data->values, num_points, function_values);

        /* scale the weights by the jacobian. */
        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }

        memcpy(function_values_scaled, function_values,
               sizeof(double)*(21*num_points));

        /*
         * scale the first set of function values by the weights and
         * determinant. Then perform matrix multiplication.
         */
        cf_diagonal_multiply(21, num_points, function_values_scaled, weights_scaled);
        DGEMM_WRAPPER_NT(i_twentyone, i_twentyone, num_points,
                         function_values_scaled, function_values, mass_matrix);
        return status;
}

/**
 * @brief Calculate the local stiffness matrix.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field. No SUPG stabilization will be done if this is NULL.
 * @param[out] stiffness_matrix Pointer to the array which will be filled in
 * with stiffness matrix data. The array must be of length (at least) 441.
 */
int cf_ap_local_stiffness(const cf_ref_arrays_s* ref_data,
                          const cf_local_element_s* local_element,
                          cf_convection_f convection_function,
                          double stiffness_matrix[static 441])
{
        int i;
        int num_points = ref_data->num_points;
        double dx[21*num_points];
        double dx_scaled[21*num_points];
        double dy[21*num_points];
        double dy_scaled[21*num_points];
        double weights_scaled[num_points];
        int status = 0;
        double C[441];
        /* These are not used, but the original ArgyrisPack code expects to
         * compute them.
         */
        double B[4];
        double b[2];
        /* stuff for DGEMM. */
        int i_twentyone = 21;

        cf_ap_physical_maps(local_element, C, B, b);
        ap_physical_gradients(C, B, ref_data->dx, ref_data->dy, num_points, dx, dy);

        /* scale the weights by the jacobian. */
        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }

        memcpy(dx_scaled, dx, sizeof(double)*(21*num_points));
        memcpy(dy_scaled, dy, sizeof(double)*(21*num_points));

        /*
         * scale the first set of gradient values by the weights and
         * determinant. Then perform matrix multiplication.
         */
        cf_diagonal_multiply(21, num_points, dx_scaled, weights_scaled);
        cf_diagonal_multiply(21, num_points, dy_scaled, weights_scaled);
        DGEMM_WRAPPER_NT(i_twentyone, i_twentyone, num_points, dx_scaled, dx,
                         stiffness_matrix);
        DGEMM_WRAPPER_NT_ADD_C(i_twentyone, i_twentyone, num_points, dy_scaled,
                               dy, stiffness_matrix);
        return status;
}

/**
 * @brief Calculate the local biharmonic matrix.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field. This is not actually used in this function, but the
 * argument is here to keep a consistent interface.
 * @param[out] biharmonic_matrix Pointer to the array which will be filled in
 * with biharmonic matrix data. The array must be of size (at least)
 * num_basis_functions**2.
 */
int cf_ap_local_biharmonic(const cf_ref_arrays_s* ref_data,
                        const cf_local_element_s* local_element,
                        cf_convection_f convection_function,
                        double biharmonic_matrix[static 441])
{
        int i;
        int num_points = ref_data->num_points;
        double dxx[21*num_points];
        double dxy[21*num_points];
        double dyy[21*num_points];
        int status = 0;
        double C[441];
        /* These are not used, but the original ArgyrisPack code expects to
         * compute them.
         */
        double B[4];
        double b[2];
        double weights_scaled[num_points];
        /* stuff for LAPACK */
        int i_twentyone = 21;

        cf_ap_physical_maps(local_element, C, B, b);
        ap_physical_hessians(C, B, ref_data->dxx, ref_data->dxy, ref_data->dyy,
                             num_points, dxx, dxy, dyy);

        /* Reassign dxx and dyy to be values of the laplacian. */
        for (i = 0; i < 21*num_points; i++) {
                dxx[i] += dyy[i];
        }
        memcpy(dyy, dxx, sizeof(double)*(21*num_points));

        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }

        /*
         * Scale one set of laplacian values by the weights (themselves scaled
         * by the Jacobian) and then calculate the matrix of inner products.
         */
        cf_diagonal_multiply(21, num_points, dxx, weights_scaled);
        DGEMM_WRAPPER_NT(i_twentyone, i_twentyone, num_points, dxx, dyy,
                         biharmonic_matrix);
        return status;
}

/**
 * @brief Calculate the local beta plane matrix.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field. This is not actually used in this function, but the
 * argument is here to keep a consistent interface.
 * @param[out] betaplane_matrix Pointer to the array which will be filled in
 * with betaplane matrix data. The array must be of length (at least) 441.
 */
int cf_ap_local_betaplane(const cf_ref_arrays_s* ref_data,
                          const cf_local_element_s* local_element,
                          cf_convection_f convection_function,
                          double betaplane_matrix[static 441])
{
        int i;
        int num_points = ref_data->num_points;
        double values[21*num_points];
        double dx[21*num_points];
        double dy[21*num_points];
        double weights_scaled[num_points];
        int status = 0;
        double C[441];
        /* These are not used, but the original ArgyrisPack code expects to
         * compute them.
         */
        double B[4];
        double b[2];
        /* stuff for LAPACK */
        int i_twentyone = 21;

        cf_ap_physical_maps(local_element, C, B, b);
        ap_physical_gradients(C, B, ref_data->dx, ref_data->dy, num_points, dx, dy);
        ap_physical_values(C, ref_data->values, num_points, values);

        /* scale the weights by the jacobian. */
        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }

        /*
         * scale the function values by the weights and determinant. Then
         * perform matrix multiplication.
         */
        cf_diagonal_multiply(21, num_points, values, weights_scaled);
        DGEMM_WRAPPER_NT(i_twentyone, i_twentyone, num_points, values, dx,
                         betaplane_matrix);
        return status;
}

/**
 * @brief Calculate the local load vector.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field. No SUPG stabilization will be done if this is NULL.
 * @param[in] forcing_function Function pointer for calculating the
 * forcing field.
 * @param[in] time Time value at which to evaluate forcing_function.
 * @param[out] local_load_vector Pointer to the array which will be filled in
 * with the load vector. Must be of length at least num_points.
 */
int cf_ap_local_load(const cf_ref_arrays_s* ref_data,
                     const cf_local_element_s* local_element,
                     cf_convection_f convection_function,
                     cf_forcing_f forcing_function, double time,
                     double local_load_vector[21])
{
        int i;
        int i_one = 1;
        int num_points = ref_data->num_points;
        int length = num_points*21;
        double force_values[num_points];
        double test_function[num_points*21];
        double untransformed_local_load_vector[21];
        int status = 0;
        double xs[num_points], ys[num_points];
        double C[441];
        /* These are not used, but the original ArgyrisPack code expects to
         * compute them.
         */
        double B[4];
        double b[2];
        /* stuff for DGEMM. */
        int i_twentyone = 21;

        cf_ap_physical_maps(local_element, C, B, b);

        cf_affine_transformation(ref_data, local_element, xs, ys);
        for (i = 0; i < num_points; i++) {
                force_values[i] = local_element->jacobian*ref_data->weights[i]
                        *forcing_function(time, xs[i], ys[i]);
        }
        if (convection_function == NULL
            || local_element->supg_stabilization_constant == 0.0) {
                memcpy(test_function, ref_data->values, sizeof(double)*length);
        }
        else {
                /* not implemented. */
                return 1;
                cf_supg_test_function_value(ref_data, local_element,
                                            convection_function,
                                            test_function);
        }

        DGEMM_WRAPPER_NT(i_twentyone, i_one, num_points,
                         test_function, force_values, untransformed_local_load_vector);
        DGEMM_WRAPPER(i_twentyone, i_one, i_twentyone, C, untransformed_local_load_vector,
                      local_load_vector);

        return status;
}

/**
 * @brief Build a mass matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_ap_build_mass(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                     cf_convection_f convection_function,
                     cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_ap_local_mass, &mesh, &ref_data,
                                         convection_function, &matrix);
        return status;
}

/**
 * @brief Build a betaplane matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_ap_build_betaplane(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                     cf_convection_f convection_function,
                     cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_ap_local_betaplane, &mesh, &ref_data,
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
int cf_ap_build_stiffness(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                          cf_convection_f convection_function,
                          cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_ap_local_stiffness, &mesh, &ref_data,
                                         convection_function, &matrix);
        return status;
}

/**
 * @brief Build a biharmonic matrix.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[out] matrix Sparse matrix in triplet format.
 */
int cf_ap_build_biharmonic(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                           cf_convection_f convection_function,
                           cf_triplet_matrix_s matrix)
{
        int status;
        status = cf_build_triplet_matrix(cf_ap_local_biharmonic, &mesh,
                                         &ref_data, convection_function,
                                         &matrix);
        return status;
}

/*
 * TODO this is really just a cut-and-paste of the Lagrange case. There should
 * be a way to have a common backend global load vector assembler for both sets
 * of elements. It is worth noting that the Argyris local load vector
 * calculation can fail but the Lagrange one will not.
 */
/**
 * @brief Build an Argyris load vector.
 *
 * @param[in] mesh Finite element mesh.
 * @param[in] ref_data Information about finite element space on the reference
 * domain.
 * @param[in] convection_function Function for calculating the convection field
 * and its derivatives.
 * @param[in] forcing_function Function for calculating the forcing term.
 * @param[in] time Time value at which to evaluate the forcing function.
 * @param[out] vector Load vector.
 */
int cf_ap_build_load(const cf_mesh_s* mesh, const cf_ref_arrays_s* ref_data,
                     cf_convection_f convection_function,
                     cf_forcing_f forcing_function, double time,
                     cf_vector_s* vector)
{
        double xs[3], ys[3];
        double local_load_vector[ref_data->num_basis_functions];
        cf_local_element_s element_data;
        int basis_num, element_num, node_index, i;
        int status = 0;

        for (element_num = 0; element_num < mesh->num_elements; element_num++) {
                cf_check(cf_get_corners(mesh, element_num, xs, ys));
                element_data = cf_init_element_data(xs, ys,
                                                    ref_data->global_supg_constant);
                /* cf_ap_local_load can fail (stabilization is not yet supported). */
                cf_check(cf_ap_local_load(
                                 ref_data, &element_data, convection_function,
                                 forcing_function, time, local_load_vector));

                for (i = 0; i < mesh->num_basis_functions; i++) {
                        node_index = CF_INDEX(element_num, i,
                                              mesh->num_elements,
                                              mesh->num_basis_functions);
                        basis_num = mesh->elements[node_index];
                        if (basis_num >= vector->length) {
                                fprintf(stderr, CF_INDEX_TOO_LARGE, basis_num,
                                        vector->length);
                                basis_num = 0;
                                status = 1;
                                cf_check(1);
                        }
                        vector->values[basis_num] += local_load_vector[i];
                }
        }
        return status;
fail:
        return status;
}
