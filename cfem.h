/** @file */
#pragma once
/**
 * @brief Macro indicating that we are using row-major storage.
 */
#define USE_ROW_MAJOR_ORDERING

/**
 * @brief Macro for accessing matrix entries in row major order.
 */
#define CF_INDEX(row, col, num_rows, num_cols) ((row)*(num_cols) + (col))

/**
 * @brief Error handling macro. Inspired by Zed Shaw's dbg.h macro set.
 */
#define cf_check(x) if(x) {goto fail;}

/**
 * @brief Error message for walking off the end of an array. Has two format
 * tags; one for the index and another for the length of the vector.
 */
#define CF_INDEX_TOO_LARGE \
        "[ERROR] Array index (%d) greater than length of array (%d)\n"

/**
 * @brief A struct containing a pointer to an array as well as its length.
 * arrays for storing a sparse matrix as triplets. May contain duplicate
 * indices.
 */
typedef struct {
        const int length;              /**< Length of the three arrays. */
        double* const restrict values; /**< Entry values. */
} cf_vector_s;

/**
 * @brief A struct containing values and derivatives of the convection field.
 */
typedef struct {
        double value[2];
        double dx[2];
        double dy[2];
} cf_convection_s;

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
        double jacobian;                    /**< Jacobian of transformation
                                               y := B x + b                 */
        double supg_stabilization_constant; /**< Streamline-Upwind
                                               Petrov-Galerkin stabilization
                                               constant.                     */
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
        double* const restrict xs; /**< Pointer to x-coordinates of quadrature
                                      points.                                 */
        double* const restrict ys; /**< Pointer to y-coordinates of quadrature
                                      points.                                 */
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
        double global_supg_constant;    /**< Global value of the SUPG
                                             stabilization constant.          */
} cf_ref_arrays_s;

/**
 * @brief Struct containing information about the quadrature rule.
 */
typedef struct {
        int num_points;                 /**< Number of quadrature points.     */
        double* const restrict weights; /**< Pointer to quadrature rule weight
                                           values.                            */
        double* const restrict points;  /**< Pointer to quadrature rule
                                           coordinates.                       */
} cf_quad_rule_s;

/**
 * @brief Function pointer for calculating the convection field.
 *
 * @param[in] x X-coordinate of point.
 * @param[in] y Y-coordinate of point.
 */
typedef cf_convection_s (*cf_convection_f)(double x, double y);

/**
 * @brief Function pointer for calculating the forcing function.
 *
 * @param[in] t time value.
 * @param[in] x X-coordinate of point.
 * @param[in] y Y-coordinate of point.
 */
typedef double (*cf_forcing_f)(double t, double x, double y);

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
typedef int (*cf_local_matrix_f)(const cf_ref_arrays_s*,
                                 const cf_local_element_s*,
                                 cf_convection_f,
                                 double* restrict);
