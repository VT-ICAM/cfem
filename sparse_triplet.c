/** @file */
/**
 * @brief Build a triplet-format sparse finite element matrix.
 *
 * @param[in] cf_local_matrix_f A function which computes local matrices.
 * @param[in] mesh A struct describing the 2D finite element mesh.
 * @param[in] ref_data A struct containing information about the finite element
 * space on the reference element.
 * @param[in] convection_function Function pointer for calculating the
 * convection field.
 * @param[out] matrix The triplet-format matrix.
 *
 * @retval status Integer of the status of the computation: zero for success,
 * 1 for illegal array access, and 2 for incorrect parameter choices.
 */
int cf_build_triplet_matrix(cf_local_matrix_f local_matrix_function,
                            const cf_mesh_s* mesh,
                            const cf_ref_arrays_s* ref_data,
                            cf_convection_f convection_function,
                            cf_triplet_matrix_s* matrix)
{
        cf_local_element_s element_data;
        int element_num;
        int local_index, global_index, node_index;
        int left_basis_num, right_basis_num;
        int i, j, status;
        double xs[3], ys[3];
        double local_matrix[ipow(mesh->num_basis_functions, 2)];

        status = 0;

        for (element_num = 0; element_num < mesh->num_elements; element_num++) {
                cf_check(cf_get_corners(mesh, element_num, xs, ys));
                element_data = cf_init_element_data(xs, ys,
                                                    ref_data->global_supg_constant);
                local_matrix_function(ref_data, &element_data, convection_function,
                                      local_matrix);

                for (i = 0; i < mesh->num_basis_functions; i++) {
                        node_index = CF_INDEX(element_num, i,
                                              mesh->num_elements,
                                              mesh->num_basis_functions);
                        left_basis_num = mesh->elements[node_index];
                        for (j = 0; j < mesh->num_basis_functions; j++) {
                                node_index = CF_INDEX(element_num, j,
                                                      mesh->num_elements,
                                                      mesh->num_basis_functions);
                                global_index = element_num*
                                        ipow(mesh->num_basis_functions, 2)
                                        + mesh->num_basis_functions*i + j;
                                if (global_index >= matrix->length) {
                                        fprintf(stderr, CF_INDEX_TOO_LARGE,
                                                global_index, matrix->length);
                                        global_index = 0;
                                        status = 1;
                                        printf("index too large!\n");
                                        cf_check(1);
                                }
                                local_index = CF_INDEX(i, j,
                                        mesh->num_basis_functions,
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
fail:
        return status;
}
