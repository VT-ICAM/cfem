/** @file */
/**
 * @brief Instantiate the physical element data given triangle corners.
 *
 * @param[in] xs The x-coordinate corners of the triangle.
 * @param[in] ys The y-coordinate corners of the triangle.
 *
 * @retval local_element A `cf_local_element_s` struct containing the corners
 * and affine mappings.
 */
cf_local_element_s cf_init_element_data(double xs[static 3],
                                        double ys[static 3],
                                        double supg_stabilization_constant)
{
        double b00, b01, b10, b11;
        b00 = xs[1] - xs[0];
        b01 = xs[2] - xs[0];
        b10 = ys[1] - ys[0];
        b11 = ys[2] - ys[0];
        cf_local_element_s local_element = {.xs = {xs[0], xs[1], xs[2]},
                                            .ys = {ys[0], ys[1], ys[2]},
                                            .B = {b00, b01, b10, b11},
                                            .b = {xs[0], ys[0]},
                                            .jacobian = fabs(b00*b11 - b01*b10),
                                            .supg_stabilization_constant =
                                            supg_stabilization_constant};
        return local_element;
}

/**
 * @brief Extract the coordinates of an element's corners from the mesh.
 *
 * @param[in] mesh Pointer to a struct containing the finite element mesh.
 * @param[in] element_num Element number (from zero).
 * @param[out] xs Pointer to a length 3 array containing the x-coordinates of
 * the corners.
 * @param[out] ys Pointer to a length 3 array containing the y-coordinates of
 * the corners.
 */
int cf_get_corners(const cf_mesh_s* mesh, int element_num, double xs[static 3],
                   double ys[static 3])
{
        int corner_num, node_num, node_index;
        int status = 0;
        int max_index = mesh->num_elements*mesh->num_basis_functions;
        for (corner_num = 0; corner_num < 3; corner_num++) {
                node_index = CF_INDEX(element_num, corner_num,
                                   mesh->num_elements,
                                   mesh->num_basis_functions);
                if (node_index >= max_index) {
                        fprintf(stderr, CF_INDEX_TOO_LARGE, node_index,
                                max_index);
                        /*
                         * In the case of openMP, it may not be possible to
                         * break out of a loop in case of failure. Hence do the
                         * 'safe' thing and populate with data at the beginning
                         * of the nodal array.
                         */
                        node_index = 0;
                        status = 1;
                }
                node_num = mesh->elements[node_index];
                xs[corner_num] = mesh->nodes[CF_INDEX(node_num, 0,
                                                      mesh->num_nodes, 2)];
                ys[corner_num] = mesh->nodes[CF_INDEX(node_num, 1,
                                                      mesh->num_nodes, 2)];
        }
        return status;
}
