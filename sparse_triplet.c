#include <stdio.h>
/* Build the sparse triplet format in the most generic way possible. */
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

/**
 * @brief Instantiate the physical element data given triangle corners.
 *
 * @param[in] xs The x-coordinate corners of the triangle.
 * @param[in] ys The y-coordinate corners of the triangle.
 *
 * @retval local_element A `cf_local_element_s` struct containing the corners
 * and affine mappings.
 */
cf_local_element_s cf_init_element_data(double* restrict xs, double* restrict ys) {
        double b00, b01, b10, b11;
        b00 = xs[1] - xs[0];
        b01 = xs[2] - xs[0];
        b10 = ys[1] - ys[0];
        b11 = ys[2] - ys[0];
        cf_local_element_s local_element = {.xs = {xs[0], xs[1], xs[2]},
                                            .ys = {ys[0], ys[1], ys[2]},
                                            .B = {b00, b01, b10, b11},
                                            .b = {xs[0], ys[0]}};

        return local_element;
}

/**
 * @brief Build a triplet-format sparse finite element matrix.
 *
 * @param[in] local_matrix_function_f A function which computes local matrices.
 * @param[in] mesh A struct describing the 2D finite element mesh.
 * @param[in] ref_data A struct containing information about the finite element
 * space on the reference element.
 * @param[out] matrix The triplet-format matrix.
 *
 * @retval status Integer of the status of the computation: zero for success,
 * -1 for illegal element access.
 */
int cf_build_triplet_matrix(local_matrix_function_f local_matrix_function,
                            const cf_mesh_s* mesh,
                            const cf_ref_arrays_s* ref_data,
                            cf_triplet_matrix_s* matrix) {
        const int max_element_array_index = mesh->num_elements
                *mesh->num_basis_functions;
        cf_local_element_s element_data;
        int element_num;
        int local_index, global_index, node_index;
        int corner_num, node_num, left_basis_num, right_basis_num;
        int i, j, status;
        double xs[3], ys[3];
        double local_matrix[ipow(mesh->num_basis_functions, 2)];

        element_data = cf_init_element_data(xs, ys);
        status = 0;

        for (element_num = 0; element_num < mesh->num_elements; element_num++) {
                for (corner_num = 0; corner_num < 3; corner_num++) {
                        node_index = INDEX(element_num, corner_num,
                                           mesh->num_elements,
                                           mesh->num_basis_functions);
                        if (node_index >= max_element_array_index) {
                                fprintf(stderr, "attempted to access"
                                        " nonexistent node with index %d\n",
                                        node_index);
                                node_index = 0;
                                status = -1;
                        }
                        node_num = mesh->elements[node_index];
                        xs[corner_num] = mesh->nodes[INDEX(node_num, 0,
                                                           mesh->num_nodes, 2)];
                        ys[corner_num] = mesh->nodes[INDEX(node_num, 1,
                                                           mesh->num_nodes, 2)];
                }
                element_data = cf_init_element_data(xs, ys);

                local_matrix_function(ref_data, &element_data, local_matrix);

                for (i = 0; i < mesh->num_basis_functions; i++) {
                        node_index = INDEX(element_num, i, mesh->num_elements,
                                           mesh->num_basis_functions);
                        left_basis_num = mesh->elements[node_index];
                        for (j = 0; j < mesh->num_basis_functions; j++) {
                                node_index = INDEX(element_num, j,
                                                   mesh->num_elements,
                                                   mesh->num_basis_functions);
                                global_index = element_num*
                                        ipow(mesh->num_basis_functions, 2)
                                        + mesh->num_basis_functions*i + j;
                                if (global_index >= matrix->length) {
                                        fprintf(stderr, "attempted to walk off "
                                                "end of data array in matrix\n");
                                        global_index = 0;
                                        status = -1;
                                }
                                local_index = INDEX(i, j, mesh->num_basis_functions,
                                                    mesh->num_basis_functions);

                                right_basis_num = mesh->elements[node_index];
                                matrix->values[global_index] =
                                        local_matrix[local_index];
                                matrix->rows[global_index] = left_basis_num;
                                matrix->columns[global_index] = right_basis_num;
                        }
                }
        }
        return status;
}
