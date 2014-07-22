/** @file */
/**
 * @brief Build a load vector.
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
int cf_build_load(const cf_mesh_s mesh, const cf_ref_arrays_s ref_data,
                  cf_convection_f convection_function,
                  cf_forcing_f forcing_function, double time,
                  cf_vector_s vector)
{
        double xs[3], ys[3];
        double local_load_vector[ref_data.num_basis_functions];
        cf_local_element_s element_data;
        int basis_num, element_num, node_index, i;
        int status = 0;

        for (element_num = 0; element_num < mesh.num_elements; element_num++) {
                cf_check(cf_get_corners(&mesh, element_num, xs, ys));
                element_data = cf_init_element_data(xs, ys,
                                                    ref_data.global_supg_constant);
                cf_local_load(&ref_data, &element_data, convection_function,
                              forcing_function, time, local_load_vector);

                for (i = 0; i < mesh.num_basis_functions; i++) {
                        node_index = CF_INDEX(element_num, i,
                                              mesh.num_elements,
                                              mesh.num_basis_functions);
                        basis_num = mesh.elements[node_index];
                        if (basis_num >= vector.length) {
                                fprintf(stderr, CF_INDEX_TOO_LARGE, basis_num,
                                        vector.length);
                                basis_num = 0;
                                status = 1;
                                cf_check(1);
                        }
                        vector.values[basis_num] += local_load_vector[i];
                }
        }
        return status;
fail:
        return status;
}
