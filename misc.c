/** @file */
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
                          double matrix[static rows*cols],
                          double diagonal[static cols]) {
        int i, j;
        /* traverse the matrix in row-major order. */
        for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) {
                        matrix[CF_INDEX(i, j, rows, cols)] *= diagonal[j];
                }
        }
}

/**
 * @brief Take the integer power of an integer.
 *
 * @param[in] base The base of the exponent.
 * @param[in] power The power of the exponent.
 *
 * @retval answer The value of base**power.
 */
int ipow(int base, int power) {
        int i;
        int answer = 1;
        for (i = 0; i < power; i++) {
                answer *= base;
        }
        return answer;
}
