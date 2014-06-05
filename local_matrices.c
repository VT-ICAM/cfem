/** @file */
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
 * mass matrix data. The array must be of size (at least)
 * num_basis_functions**2.
 */
int cf_local_mass(const cf_ref_arrays_s* ref_data,
                  const cf_local_element_s* local_element,
                  cf_convection_f convection_function,
                  double mass_matrix[
                          static ipow(ref_data->num_basis_functions, 2)])
{
        int i;
        int num_points = ref_data->num_points;
        int num_basis_functions = ref_data->num_basis_functions;
        int length = num_points*num_basis_functions;
        double supg_test_function_values[length];
        double values[length];
        double weights_scaled[num_points];

        int status = 0;

        memcpy(values, ref_data->values, sizeof(double)*length);

        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }

        cf_diagonal_multiply(num_basis_functions, num_points, values,
                             weights_scaled);

        if (convection_function == NULL
            || local_element->supg_stabilization_constant == 0.0) {
                DGEMM_WRAPPER_NT(num_basis_functions, num_basis_functions,
                                 num_points, ref_data->values, values,
                                 mass_matrix);
        }
        else {
                cf_supg_test_function_value(ref_data, local_element,
                                            convection_function,
                                            supg_test_function_values);
                DGEMM_WRAPPER_NT(num_basis_functions, num_basis_functions,
                                 num_points, supg_test_function_values, values,
                                 mass_matrix);
        }
        return status;
}

/**
 * @brief Calculate the local convection matrix.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field. No SUPG stabilization will be done if this is NULL.
 * @param[out] convection_matrix Pointer to the array which will be filled in
 * with convection matrix data. The array must be of size (at least)
 * num_basis_functions**2.
 */
int cf_local_convection(const cf_ref_arrays_s* ref_data,
                        const cf_local_element_s* local_element,
                        cf_convection_f convection_function,
                        double convection_matrix[
                                static ipow(ref_data->num_basis_functions, 2)])
{
        int i;
        int num_points = ref_data->num_points;
        int num_basis_functions = ref_data->num_basis_functions;
        int length = num_points*num_basis_functions;
        int point_num, basis_num, index;
        double dx[length];
        double dy[length];
        double xs[length];
        double ys[length];
        double test_function_values[length];
        double weights_scaled[num_points];
        cf_convection_s convection;

        int status = 0;

        cf_affine_transformation(ref_data, local_element, xs, ys);
        cf_physical_gradients(ref_data, local_element, dx, dy);
        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }

        if (convection_function == NULL
            || local_element->supg_stabilization_constant == 0.0) {
                memcpy(test_function_values, ref_data->values,
                       sizeof(double)*length);
        }
        else {
                cf_supg_test_function_value(ref_data, local_element,
                                            convection_function,
                                            test_function_values);
        }

        for (point_num = 0; point_num < num_points; point_num++) {
                convection = convection_function(xs[point_num], ys[point_num]);
                for (basis_num = 0; basis_num < num_basis_functions;
                     basis_num++) {
                        index = CF_INDEX(basis_num, point_num,
                                         num_basis_functions, num_points);
                        dx[index] *= convection.value[0];
                        dx[index] += convection.value[1]*dy[index];
                }
        }
        cf_diagonal_multiply(num_basis_functions, num_points, dx,
                             weights_scaled);

        DGEMM_WRAPPER_NT(num_basis_functions, num_basis_functions,
                         num_points, test_function_values, dx,
                         convection_matrix);
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
 * with stiffness matrix data. The array must be of size (at least)
 * num_basis_functions**2.
 */
int cf_local_stiffness(const cf_ref_arrays_s* ref_data,
                       const cf_local_element_s* local_element,
                       cf_convection_f convection_function,
                       double stiffness_matrix[
                               static ipow(ref_data->num_basis_functions, 2)])
{
        int i;
        int num_points = ref_data->num_points;
        int num_basis_functions = ref_data->num_basis_functions;
        int length = num_points*num_basis_functions;
        double dx[length];
        double dy[length];
        double test_function_dx[length];
        double test_function_dy[length];
        double weights_scaled[num_points];

        int status = 0;

        cf_physical_gradients(ref_data, local_element, dx, dy);
        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }
        if (convection_function == NULL
            || local_element->supg_stabilization_constant == 0.0) {
                memcpy(test_function_dx, dx, sizeof(double)*length);
                memcpy(test_function_dy, dy, sizeof(double)*length);
        }
        else {
                cf_supg_test_function_gradient(ref_data, local_element,
                                               convection_function,
                                               test_function_dx,
                                               test_function_dy);
        }
        cf_diagonal_multiply(num_basis_functions, num_points, dx,
                             weights_scaled);
        cf_diagonal_multiply(num_basis_functions, num_points, dy,
                             weights_scaled);

        DGEMM_WRAPPER_NT(num_basis_functions, num_basis_functions,
                         num_points, test_function_dx, dx, stiffness_matrix);
        DGEMM_WRAPPER_NT_ADD_C(num_basis_functions, num_basis_functions,
                               num_points, test_function_dy, dy,
                               stiffness_matrix);
        return status;
}

/**
 * @brief Calculate the local hessian matrix.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field. This is not actually used in this function, but the
 * argument is here to keep a consistent interface.
 * @param[out] stiffness_matrix Pointer to the array which will be filled in
 * with stiffness matrix data. The array must be of size (at least)
 * num_basis_functions**2.
 */
int cf_local_hessian(const cf_ref_arrays_s* ref_data,
                     const cf_local_element_s* local_element,
                     cf_convection_f convection_function,
                     double hessian_matrix[
                             static ipow(ref_data->num_basis_functions, 2)])
{
        int i;
        int num_points = ref_data->num_points;
        int num_basis_functions = ref_data->num_basis_functions;
        double dxx[num_basis_functions*num_points];
        double dxy[num_basis_functions*num_points];
        double dyy[num_basis_functions*num_points];
        double weights_scaled[num_points];

        int status = 0;

        cf_physical_hessian(ref_data, local_element, dxx, dxy, dyy);

        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_data->weights[i]*local_element->jacobian;
        }
        /* Reassign dxx and dyy to be values of the laplacian. */
        for (i = 0; i < num_basis_functions*num_points; i++) {
                dxx[i] += dyy[i];
        }
        memcpy(dyy, dxx, sizeof(double)*(num_basis_functions*num_points));

        /*
         * Scale one set of laplacian values by the weights (themselves scaled
         * by the Jacobian) and then calculate the matrix of inner products.
         */
        cf_diagonal_multiply(num_basis_functions, num_points, dxx,
                             weights_scaled);
        DGEMM_WRAPPER_NT(num_basis_functions, num_basis_functions, num_points,
                         dxx, dyy, hessian_matrix);
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
int cf_local_load(const cf_ref_arrays_s* ref_data,
                  const cf_local_element_s* local_element,
                  cf_convection_f convection_function,
                  cf_forcing_f forcing_function, double time,
                  double local_load_vector[static ref_data->num_points])
{
        int i;
        int i_one = 1;
        int num_points = ref_data->num_points;
        int num_basis_functions = ref_data->num_basis_functions;
        int length = num_points*num_basis_functions;
        double force_values[num_points];
        double test_function[num_points*num_basis_functions];
        int status = 0;

        double xs[num_points], ys[num_points];

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
                cf_supg_test_function_value(ref_data, local_element,
                                            convection_function,
                                            test_function);
        }

        DGEMM_WRAPPER_NT(num_basis_functions, i_one, num_points,
                         test_function, force_values, local_load_vector);

        return status;
}
