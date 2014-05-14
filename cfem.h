/** @file */
#pragma once
/**
 * @brief Macro indicating that we are using row-major storage.
 */
#define USE_ROW_MAJOR_ORDERING

/**
 * @brief Macro for accessing matrix entries in row major order.
 */
#define INDEX(row, col, num_rows, num_cols) ((row)*(num_cols) + (col))

/**
 * @brief A struct containing pointers to the three usual (row, column value)
 * arrays for storing a sparse matrix as triplets. May contain duplicate
 * indices.
 */
typedef struct {
        const int length;              /**< Length of the three arrays. */
        int* const restrict rows;      /**< Row indices. */
        int* const restrict columns;   /**< Column indices. */
        double* const restrict values; /**< Entry values. */
} cf_triplet_matrix_s;

/**
 * @brief A struct containing information about a physical element.
 */
typedef struct {
        double xs[3];                       /**< The corner x-coordinates.  */
        double ys[3];                       /**< The corner y-coordinates.  */
        double B[4];                        /**< Affine mapping multiplier. */
        double b[2];                        /**< Affine mapping offset.     */
        double supg_stabilization_constant; /**< Streamline-Upwind
                                               Petrov-Galerkin stabilization
                                               constant.*/
} cf_local_element_s;

/**
 * @brief Struct containing information about a finite element mesh.
 */
typedef struct {
        int num_nodes;                /**< Number of nodes in the mesh.       */
        double* const restrict nodes; /**< Pointer to the node data array.    */
        int num_elements;             /**< Number of elements in the mesh.    */
        int num_basis_functions;      /**< Number of basis functions per
                                         element.                             */
        int* const restrict elements; /**< Pointer to the element data array. */
} cf_mesh_s;

/**
 * @brief Struct containing information about the finite element space.
 */
typedef struct {
        int num_points;                 /**< Number of quadrature points.     */
        int num_basis_functions;        /**< Number of basis functions.       */
        double* const restrict weights; /**< Pointer to quadrature weight
                                           values.                            */
        double* const restrict values;  /**< Pointer to values of reference
                                           basis functions.                   */
        double* const restrict dx;      /**< Pointer to values of x-derivatives
                                           of basis functions.*/
        double* const restrict dy;      /**< Pointer to values of y-derivatives
                                           of basis functions.                */
        double* const restrict dxx;     /**< Pointer to values of xx-derivatives
                                           of basis functions.                */
        double* const restrict dxy;     /**< Pointer to values of xy-derivatives
                                           of basis functions.                */
        double* const restrict dyy;     /**< Pointer to values of yy-derivatives
                                           of basis functions.                */
} cf_ref_arrays_s;

/**
 * @brief Function pointer that provides a common calling list to each of the
 * local matrix calculations.
 *
 * @param[in] ref_data Struct containing information about the reference
 * element.
 * @param[in] local_element Struct containing information about the current
 * element.
 * @param[out] matrix The array which will be populated as output.
 */
typedef void (*local_matrix_function_f)(const cf_ref_arrays_s*,
                                        const cf_local_element_s*, double*);
