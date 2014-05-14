/** @file */
/**
   @brief DGEMM prototype for C.
 */
void dgemm_(char*, char*, const int*, const int*, const int*, const double* const,
           const double* const,
           const int*, const double* const, const int*, double*, double*, const int*);

/**
   @brief Constant for char 'N' by reference.
 */
char __c_N = 'N';

/**
   @brief Constant for char 'T' by reference.
 */
char __c_T = 'T';

/**
   @brief Constant for passing double precision value 1d0 by reference.
 */
double __d_zero = 0;

/**
   @brief Constant for passing double precision value 0d0 by reference.
 */
double __d_one = 1;

/**
 * @brief Wrapper around DGEMM, for C := C + A*B.T.
 *
 * @param m Number of rows in A.
 * @param n Number of columns in B-transpose.
 * @param k Number of rows in B-transpose (and columns in A).
 * @param A First matrix.
 * @param B Second matrix (algorithm works with its transpose).
 * @param C Output matrix.
 */
#define DGEMM_WRAPPER_NT(m, n, k, A, B, C) \
dgemm_(&(__c_T), &(__c_N), &(n), &(m), &(k), &(__d_one), B, &(k), A, \
      &(k), &(__d_zero), C, &(n))

/**
 * @brief Wrapper around DGEMM, for A := A + B*E.T + A.
 *
 * @param m Number of rows in B.
 * @param n Number of columns in E-transpose.
 * @param k Number of rows in E-transpose (and columns in B).
 * @param B First matrix.
 * @param E Second matrix (algorithm works with its transpose).
 * @param A output matrix.
 */
#define DGEMM_WRAPPER_NT_ADD_C(m, n, k, A, B, C) \
dgemm_(&(__c_T), &(__c_N), &(n), &(m), &(k), &(__d_one), B, &(k), A, \
       &(k), &(__d_one), C, &(n))

/**
 * @brief Multiply a matrix by a diagonal matrix (represented as a flat array)
 * in place on the right.
 *
 * @param[in] rows Number of rows in the matrix.
 * @param[in] cols Number of columns in the matrix.
 * @param[out] matrix The matrix. It will be multiplied in-place on the right by
 * the diagonal matrix.
 * @param[in] diagonal The diagonal matrix, represented as a 1D array.
 */
void cf_diagonal_multiply(const int rows, const int cols,
                          double* restrict matrix,
                          double* restrict diagonal) {
        int i, j;
        /* traverse the matrix in row-major order. */
        for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) {
                        matrix[INDEX(i, j, rows, cols)] *= diagonal[j];
                }
        }
}

/**
 * @brief Calculate the local mass matrix.
 *
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[out] mass_matrix Pointer to the array which will be filled in with
 * mass matrix data. The array must be of size (at least)
 * num_basis_functions**2.
 */
void cf_local_mass(const cf_ref_arrays_s* ref_arrays,
                   const cf_local_element_s* local_element,
                   double* restrict mass_matrix)
{
        int i;
        int num_points = ref_arrays->num_points;
        int num_basis_functions = ref_arrays->num_basis_functions;
        int length = num_points*num_basis_functions;
        double values[length];
        double weights_scaled[num_points];

        const double* B = local_element->B;
        const double jacobian = fabs(
                B[INDEX(0, 0, 2, 2)]*B[INDEX(1, 1, 2, 2)]
                - B[INDEX(0, 1, 2, 2)]*B[INDEX(1, 0, 2, 2)]);

        memcpy(values, ref_arrays->values, sizeof(double)*length);

        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_arrays->weights[i]*jacobian;
        }

        /*
         * scale the function values by the weights and determinant. Then
         * perform matrix multiplication.
         */
        cf_diagonal_multiply(num_basis_functions, num_points, values,
                             weights_scaled);

        DGEMM_WRAPPER_NT(num_basis_functions, num_basis_functions, num_points,
                         ref_arrays->values, values, mass_matrix);
}

/**
 * @brief Calculate the local stiffness matrix.
 *
 * @param[in] local_element Pointer to the struct containing information about
 * the current element.
 * @param[out] stiffness_matrix Pointer to the array which will be filled in
 * with stiffness matrix data. The array must be of size (at least)
 * num_basis_functions**2.
 */
void cf_local_stiffness(const cf_ref_arrays_s* ref_arrays,
                        const cf_local_element_s* local_element,
                        double* restrict stiffness_matrix)
{
        int i;
        int num_points = ref_arrays->num_points;
        int num_basis_functions = ref_arrays->num_basis_functions;
        int length = num_points*num_basis_functions;
        double dx[length];
        double dy[length];
        double weights_scaled[num_points];

        const double* B = local_element->B;
        const double jacobian = fabs(
                B[INDEX(0, 0, 2, 2)]*B[INDEX(1, 1, 2, 2)]
                - B[INDEX(0, 1, 2, 2)]*B[INDEX(1, 0, 2, 2)]);

        memcpy(dx, ref_arrays->dx, sizeof(double)*length);
        memcpy(dy, ref_arrays->dy, sizeof(double)*length);

        for (i = 0; i < num_points; i++) {
                weights_scaled[i] = ref_arrays->weights[i]*jacobian;
        }

        /*
         * scale the function values by the weights and determinant. Then
         * perform matrix multiplication.
         */
        cf_diagonal_multiply(num_basis_functions, num_points, dx, weights_scaled);
        cf_diagonal_multiply(num_basis_functions, num_points, dy, weights_scaled);

        DGEMM_WRAPPER_NT(num_basis_functions, num_basis_functions, num_points,
                         ref_arrays->dx, dx, stiffness_matrix);
        DGEMM_WRAPPER_NT_ADD_C(num_basis_functions, num_basis_functions,
                               num_points, ref_arrays->dy, dx, stiffness_matrix);
}
