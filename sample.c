cf_convection_s cf_standard_cosinesine_convection(double x, double y)
{
        cf_convection_s convection = {.value = {cos(M_PI/3.0), sin(M_PI/3.0)},
                                      .dx = {0.0, 0.0},
                                      .dy = {0.0, 0.0},
        };
        return convection;
}

cf_convection_s cf_solenoidal_convection(double x, double y)
{
        cf_convection_s convection = {.value = {-y, x},
                                      .dx = {0.0, 1.0},
                                      .dy = {-1.0, 0.0},
        };
        return convection;
}


double cf_standard_forcing(double t, double x, double y)
{
        return 1.0;
}
