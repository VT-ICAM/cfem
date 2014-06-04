/** @file */

/* Save a ton of typing. */
#define CF_PHYSICAL_DATA_SIZE (ref_data->num_points*ref_data->num_basis_functions)

/**
 * @brief Transform reference coordinates to physical coordinates.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[out] physical_x X-coordinates on the physical element.
 * @param[out] physical_y Y-coordinates on the physical element.
 */
void cf_affine_transformation(const cf_ref_arrays_s* ref_data,
                              const cf_local_element_s* local_element,
                              double physical_x[static ref_data->num_points],
                              double physical_y[static ref_data->num_points])
{
        int i;
        const double* B = local_element->B;
        const double* b = local_element->b;
        for (i = 0; i < ref_data->num_points; i++) {
                physical_x[i] = B[CF_INDEX(0, 0, 2, 2)]*ref_data->xs[i]
                              + B[CF_INDEX(0, 1, 2, 2)]*ref_data->ys[i] + b[0];
                physical_y[i] = B[CF_INDEX(1, 0, 2, 2)]*ref_data->xs[i]
                              + B[CF_INDEX(1, 1, 2, 2)]*ref_data->ys[i] + b[1];
        }
}

/**
 * @brief Calculate the physical values of the test function gradient.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field.
 * @param[out] dx Values of the x-derivatives on the physical element.
 * @param[out] dy Values of the y-derivatives on the physical element.
 */
void cf_physical_gradients(const cf_ref_arrays_s* ref_data,
                           const cf_local_element_s* local_element,
                           double dx[static CF_PHYSICAL_DATA_SIZE],
                           double dy[static CF_PHYSICAL_DATA_SIZE])
{
        int i;
        int num_basis_functions = ref_data->num_basis_functions;
        int num_points = ref_data->num_points;

        const double* B = local_element->B;
        double B_det_inv = 1/(B[CF_INDEX(0, 0, 2, 2)]*B[CF_INDEX(1, 1, 2, 2)] -
                              B[CF_INDEX(0, 1, 2, 2)]*B[CF_INDEX(1, 0, 2, 2)]);

        double B_inv00 = B_det_inv*B[CF_INDEX(1, 1, 2, 2)];
        double B_inv01 = -B_det_inv*B[CF_INDEX(0, 1, 2, 2)];
        double B_inv10 = -B_det_inv*B[CF_INDEX(1, 0, 2, 2)];
        double B_inv11 = B_det_inv*B[CF_INDEX(0, 0, 2, 2)];

        /*
         * Perform the transformation using B inverse. This is equivalent to
         * putting the reference values in long columns side-by-side and
         * multiplying by B_inv.
         */
        for (i = 0; i < num_basis_functions*num_points; i++) {
                dx[i] = B_inv00*ref_data->dx[i] + B_inv10*ref_data->dy[i];
                dy[i] = B_inv01*ref_data->dx[i] + B_inv11*ref_data->dy[i];
        }
}

/**
 * @brief Calculate the physical values of the second derivatives.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[out] dxx Second x-derivative.
 * @param[out] dxy Mixed second derivative.
 * @param[out] dyy Second y-derivative.
 */
void cf_physical_hessian(const cf_ref_arrays_s* ref_data,
                         const cf_local_element_s* local_element,
                         double dxx[static CF_PHYSICAL_DATA_SIZE],
                         double dxy[static CF_PHYSICAL_DATA_SIZE],
                         double dyy[static CF_PHYSICAL_DATA_SIZE])
{
        int i;
        int num_basis_functions = ref_data->num_basis_functions;
        int num_points = ref_data->num_points;
        const double* B = local_element->B;

        /*
         * There is an extra 3x3 matrix (corresponding, in Dominguez's notation,
         * to Theta transpose) due to application of the chain rule. It's
         * entries depend on the original affine transformation from reference
         * to physical coordinates (hence the dependence on B).
         */
        double t = (B[CF_INDEX(0, 0, 2, 2)]*B[CF_INDEX(1, 1, 2, 2)]
                  - B[CF_INDEX(0, 1, 2, 2)]*B[CF_INDEX(1, 0, 2, 2)])*
                   (B[CF_INDEX(0, 0, 2, 2)]*B[CF_INDEX(1, 1, 2, 2)]
                  - B[CF_INDEX(0, 1, 2, 2)]*B[CF_INDEX(1, 0, 2, 2)]);

        double map00 = B[CF_INDEX(1, 1, 2, 2)]*B[CF_INDEX(1, 1, 2, 2)]/t;
        double map01 = -2.0*B[CF_INDEX(1, 0, 2, 2)]*B[CF_INDEX(1, 1, 2, 2)]/t;
        double map02 = B[CF_INDEX(1, 0, 2, 2)]*B[CF_INDEX(1, 0, 2, 2)]/t;

        double map10 = -1.0*B[CF_INDEX(0, 1, 2, 2)]*B[CF_INDEX(1, 1, 2, 2)]/t;
        double map11 = (B[CF_INDEX(0, 0, 2, 2)]*B[CF_INDEX(1, 1, 2, 2)]
                      + B[CF_INDEX(0, 1, 2, 2)]*B[CF_INDEX(1, 0, 2, 2)])/t;
        double map12 = -1.0*B[CF_INDEX(0, 0, 2, 2)]*B[CF_INDEX(1, 0, 2, 2)]/t;

        double map20 = B[CF_INDEX(0, 1, 2, 2)]*B[CF_INDEX(0, 1, 2, 2)]/t;
        double map21 = -2.0*B[CF_INDEX(0, 0, 2, 2)]*B[CF_INDEX(0, 1, 2, 2)]/t;
        double map22 = B[CF_INDEX(0, 0, 2, 2)]*B[CF_INDEX(0, 0, 2, 2)]/t;

        /*
         * Perform the transformation. This is equivalent to putting the
         * reference values in long columns side-by-side and multiplying by
         * (Theta inverse) transpose.
         */
        for (i = 0; i < num_points*num_basis_functions; i++) {
                dxx[i] = ref_data->dxx[i]*map00 + ref_data->dxy[i]*map01
                        + ref_data->dyy[i]*map02;
                dxy[i] = ref_data->dxx[i]*map10 + ref_data->dxy[i]*map11
                        + ref_data->dyy[i]*map12;
                dyy[i] = ref_data->dxx[i]*map20 + ref_data->dxy[i]*map21
                        + ref_data->dyy[i]*map22;
        }
}

/**
 * @brief Calculate the local values of the SUPG test function.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field.
 * @param[out] values Values of the SUPG test function.
 */
void cf_supg_test_function_value(const cf_ref_arrays_s* ref_data,
                                 const cf_local_element_s* local_element,
                                 cf_convection_f convection_function,
                                 double values[static CF_PHYSICAL_DATA_SIZE])
{
        int point_num, basis_num, index;
        int num_points = ref_data->num_points;
        int num_basis_functions = ref_data->num_basis_functions;
        int length = num_points*num_basis_functions;
        cf_convection_s convection;
        double dx[length];
        double dy[length];
        double xs[num_points];
        double ys[num_points];

        cf_affine_transformation(ref_data, local_element, xs, ys);
        cf_physical_gradients(ref_data, local_element, dx, dy);

        for (point_num = 0; point_num < num_points; point_num++) {
                convection = convection_function(xs[point_num], ys[point_num]);
                for (basis_num = 0; basis_num < num_basis_functions;
                     basis_num++) {
                        index = CF_INDEX(basis_num, point_num,
                                         num_basis_functions, num_points);
                        values[index] = ref_data->values[index]
                                + local_element->supg_stabilization_constant*
                                (convection.value[0]*dx[index]
                                 + convection.value[1]*dy[index]);
                }
        }
}

/**
 * @brief Calculate the local gradient of the SUPG test function.
 *
 * @param[in] ref_data Pointer to the struct containing information about the
 * finite element space.
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field.
 * @param[out] dx X-derivative of SUPG test function.
 * @param[out] dy Y-derivative of SUPG test function.
 */
void cf_supg_test_function_gradient(const cf_ref_arrays_s* ref_data,
                                    const cf_local_element_s* local_element,
                                    cf_convection_f convection_function,
                                    double* restrict dx,
                                    double* restrict dy)
{
        int point_num, basis_num, index;
        int num_points = ref_data->num_points;
        int num_basis_functions = ref_data->num_basis_functions;
        int length = num_points*num_basis_functions;
        cf_convection_s convection;
        double basis_dx[length];
        double basis_dy[length];
        double basis_dxx[length];
        double basis_dxy[length];
        double basis_dyy[length];
        double xs[num_points];
        double ys[num_points];

        cf_affine_transformation(ref_data, local_element, xs, ys);
        cf_physical_gradients(ref_data, local_element, basis_dx, basis_dy);
        cf_physical_hessian(ref_data, local_element, basis_dxx, basis_dxy,
                            basis_dyy);

        for (point_num = 0; point_num < ref_data->num_points; point_num++) {
                convection = convection_function(xs[point_num], ys[point_num]);
                for (basis_num = 0; basis_num < ref_data->num_basis_functions;
                     basis_num++) {
                        index = CF_INDEX(basis_num, point_num,
                                         num_basis_functions, num_points);
                        dx[index] = basis_dx[index]
                                + local_element->supg_stabilization_constant*(
                                + convection.dx[0]*basis_dx[index]
                                + convection.value[0]*basis_dxx[index]
                                + convection.dx[1]*basis_dy[index]
                                + convection.value[1]*basis_dxy[index]);
                        dy[index] = basis_dy[index]
                                + local_element->supg_stabilization_constant*(
                                + convection.dy[0]*basis_dx[index]
                                + convection.value[0]*basis_dxy[index]
                                + convection.dy[1]*basis_dy[index]
                                + convection.value[1]*basis_dyy[index]);
                }
        }
}

#undef CF_PHYSICAL_DATA_SIZE
