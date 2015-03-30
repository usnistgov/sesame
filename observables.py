from numpy import exp

def get_n(efn, v, params):
    bl = params.bl
    return exp(-bl+efn+v)

def get_p(efp, v, params):
    bl = params.bl
    eg = params.eg
    return exp(-eg+bl+efp-v)

def get_rr(efn, efp, v, tau, params):
    bl = params.bl
    eg = params.eg

    r = 1./tau*(exp(-bl+efn+v)*exp(bl-eg+efp-v) - exp(-eg))/\
        (exp(-bl+efn+v) + exp(bl-eg+efp-v) + 2*exp(-eg/2.))

    return r

def get_jn(efn, efnp1, v, vp1, dx, params):
    bl = params.bl

    if abs(v-vp1) > 1e-8:
        jn = (v - vp1)/dx * exp(-bl) * (exp(efnp1) - exp(efn))\
             / (exp(-vp1) - exp(-v))
    else:
        jn = exp(v)/dx * exp(-bl) * (exp(efnp1) - exp(efn))

    return jn

def get_jp(efp, efpp1, v, vp1, dx, params):
    bl = params.bl
    eg = params.eg

    if abs(v-vp1) > 1e-8:
        jp = - (vp1 - v)/dx * exp(-eg + bl) * (exp(efpp1) - exp(efp))\
             / (exp(vp1) - exp(v))
    else:
        jp = - exp(-v)/dx * exp(-eg + bl) * (exp(efpp1) - exp(efp))

    return jp

def get_jn_derivs(efn_i, efn_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl

    if abs(v_i-v_ip1) > 1e-8:
        defn_i = 1./dx_i * exp(-bl+efn_i) * (v_ip1 - v_i)\
                 / (exp(-v_ip1) - exp(-v_i))
        defn_ip1 = -1./dx_i * exp(-bl+efn_ip1) * (v_ip1 - v_i)\
                   / (exp(-v_ip1) - exp(-v_i))
        dv_i = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
               * (exp(-v_i) * (1+v_i-v_ip1) - exp(-v_ip1))\
               / (exp(-v_ip1) - exp(-v_i))**2 
        dv_ip1 = -1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
                * (exp(-v_ip1) * (1+v_ip1-v_i) - exp(-v_i))\
                / (exp(-v_ip1) - exp(-v_i))**2 
    else:
        defn_i = -1./dx_i * exp(-bl+efn_i) * exp(v_i)
        defn_ip1 = 1./dx_i * exp(-bl+efn_ip1) * exp(v_i)
        dv_i = 1./dx_i * exp(-bl) * (exp(efn_ip1) - exp(efn_i))\
               * exp(v_i) / 2.
        dv_ip1 = dv_i

    return defn_i, defn_ip1, dv_i, dv_ip1   

def get_jp_derivs(efp_i, efp_ip1, v_i, v_ip1, dx_i, params):
    bl = params.bl
    eg = params.eg

    if abs(v_i-v_ip1) > 1e-8:
        defp_i = 1/dx_i * exp(bl-eg+efp_i) * (v_ip1 - v_i)\
                 / (exp(v_ip1) - exp(v_i))
        defp_ip1 = -1/dx_i * exp(bl-eg+efp_ip1) * (v_ip1 - v_i)\
                   / (exp(v_ip1) - exp(v_i))
        dv_i = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
               * (exp(v_i) * (1+v_ip1-v_i) - exp(v_ip1))\
               / (exp(v_ip1) - exp(v_i))**2
        dv_ip1 = -1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
                * (exp(v_ip1) * (1-v_ip1+v_i) - exp(v_i))\
                / (exp(v_ip1)-exp(v_i))**2
        
    else:
        defp_i = 1/dx_i * exp(bl-eg+efp_i) * exp(-v_i)
        defp_ip1 = -1/dx_i * exp(bl-eg+efp_ip1) * exp(-v_i)
        dv_i = 1/dx_i * exp(bl-eg) * (exp(efp_ip1) - exp(efp_i))\
               * exp(-v_i) / 2.
        dv_ip1 = dv_i
        
    return defp_i, defp_ip1, dv_i, dv_ip1

def get_rr_derivs(efn_i, efp_i, v_i, tau, params):
    bl = params.bl
    eg = params.eg

    defp_i = 1/tau * exp(bl+efp_i+v_i)*(exp(bl)+exp(eg/2.+efn_i+v_i))**2\
             / (exp(2.*bl+efp_i)+2.*exp(bl+eg/2.+v_i)+exp(eg+efn_i+2.*v_i))**2
    defn_i = 1/tau * exp(bl+efn_i+v_i)*(exp(bl+efp_i)+exp(eg/2.+v_i))**2\
             / (exp(2.*bl+efp_i)+2.*exp(bl+eg/2.+v_i)+exp(eg+efn_i+2.*v_i))**2
    dv_i = 1/tau * exp(bl+v_i)*(-1+exp(efn_i+efp_i))\
           * (exp(2.*bl+efp_i)-exp(eg+efn_i+2.*v_i))\
           / (exp(2.*bl+efp_i)+2.*exp(bl+eg/2.+v_i)+exp(eg+efn_i+2.*v_i))**2

    return defp_i, defn_i, dv_i
