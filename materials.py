def define_material(Nc, Nv, Eg, epsilon, mu_e, mu_h, tau_e, tau_h,
                    RCenergy):
    """
    Return a dictionary with the material parameters

    Arguments
    ---------
    Nc, Nv: conduction, valence band effective densities [m^-3]
    Eg: gap [eV]
    epsilon: relative permittivity [no unit]
    mu_x: mobility for e/h [m^2 / (V.s)]
    tau_x: lifetime for e/h [s]
    RCenergy: energy of the recombination centers defined from midgap [eV]
    """
    mat = {'Nc':Nc, 'Nv':Nv, 'Eg':Eg, 'epsilon':epsilon,
           'mu_e':mu_e, 'mu_h':mu_h, 'tau_e':tau_e, 'tau_h':tau_h,
           'RCenergy':RCenergy}
    return mat

Si = {'Nc':1, 'Nv':1, 'Eg':1, 'epsilon':1,
      'mu_e':1, 'mu_p':1, 'tau_e':1, 'tau_p':1, 'RCenergy':1}

