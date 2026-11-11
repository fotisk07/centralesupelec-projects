# Control of acoustic wave by damping on a surface
# Arthur Jacquin

from math import log10, cos, pi
from cmath import sqrt, exp, cosh, sinh
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

# CONSTANTES PHYSIQUES
c_0 = 340
rho_0 = 1.29
gamma_p = 7/5

# PARAMETRES
A = B = 1
l = 3
L = 10  # choisi tres grand (10 metres)
N = 10
DEBUG = False
folder = 'C:\\Users\\arthu\\Downloads\\projet\\figures\\'

# SIGNAUX
B_0 = 1
P_0 = 3
omega_1, omega_2 = 10000, 15000
eps = 0.3
l_bande = l/10
G_0 = 1
l_0 = l/10

# MATERIAUX
melamine_foam = {
    'porosity': 0.99,  # \phi
    'resistivity': 14000,  # \sigma
    'tortuosity': 1.02,  # \alpha_h
}
isorel = {
    'porosity': 0.70,
    'resistivity': 142300,
    'tortuosity': 1.15,
}
itfh = {
    'porosity': 0.94,
    'resistivity': 9067,
    'tortuosity': 1,
}
b5 = {
    'porosity': 0.2,
    'resistivity': 2124000,
    'tortuosity': 1.22,
}


def g_1(y, omega):
    return B_0 + P_0*(omega_1 <= omega <= omega_2)


def g_2(y, omega):
    return B_0 + P_0*(1 + eps*(((y/l_bande) % 1) < .5))*(omega_1 <= omega <= omega_2)


def g_3(y, omega):
    return G_0*cos(y/l_0)


def compute_gas_constants(c_0):
    xi = 1/(c_0**2)
    a = 0
    eta = 1
    return xi, a, eta


def compute_material_constants(m, c_0, rho_0, gamma_p):
    xi = m['porosity']*gamma_p/(c_0**2)
    a = m['resistivity']*(m['porosity']**2)*gamma_p / \
        ((c_0**2)*rho_0*m['tortuosity'])
    eta = m['porosity']/m['tortuosity']
    return xi, a, eta


material = melamine_foam
xi_0, _, eta_0 = compute_gas_constants(c_0)
xi_1, a, eta_1 = compute_material_constants(material, c_0, rho_0, gamma_p)


def fourier(g, omega, k):
    ''' Coefficient k non normalise de la transformee de Fourier de g '''
    re, _ = integrate.quad(lambda y: (
        g(y, omega)*exp(-1j*pi*k*y/l)).real, -l, l)
    im, _ = integrate.quad(lambda y: (
        g(y, omega)*exp(-1j*pi*k*y/l)).imag, -l, l)
    return complex(re, im)


def e(alpha, g, omega, N):
    ''' Fonction d'erreur '''
    res = 0
    for n in range(-N, N + 1):
        k = n*pi/L
        X0 = k**2 - (xi_0/eta_0) * omega**2
        lambda_0 = sqrt(X0) if X0 >= 0 else 1j*sqrt(-X0)
        lambda_eta_0 = lambda_0 * eta_0
        F0 = 2*lambda_eta_0*cosh(lambda_0*L)
        F1 = 2*sinh(lambda_0*L)
        def f(x): return F0 + x*F1
        X1 = k**2 - (xi_1/eta_1) * omega**2
        M = sqrt(X1**2 + (a*omega/eta_1)**2)
        lambda_1 = complex(sqrt(X1 + M), sqrt(-X1 + M))/sqrt(2)
        lambda_eta_1 = lambda_1 * eta_1
        g_k = fourier(g, omega, k)
        chi = g_k * ((lambda_eta_0 - lambda_eta_1) /
                     f(lambda_eta_1) - (lambda_eta_0 - alpha)/f(alpha))
        module_carre_chi = abs(chi)**2
        gamma = g_k * ((lambda_eta_0 + lambda_eta_1) /
                       f(lambda_eta_1) - (lambda_eta_0 + alpha)/f(alpha))
        module_carre_gamma = abs(gamma)**2
        chi_gamma_conj = chi * gamma.conjugate()
        P0 = 1 - exp(-2*lambda_0*L)
        if (X0 >= 0):
            P1 = exp(2*lambda_0*L) - 1
            P2 = module_carre_chi*P0 + module_carre_gamma*P1
            ek = (A + B*k**2)*(P2/(2*lambda_0) + 2*L*chi_gamma_conj.real) + \
                B*lambda_0*P2/2 - 2*B*(lambda_0**2)*L*chi_gamma_conj.real
        else:
            P1 = module_carre_chi + module_carre_gamma
            P2 = (chi_gamma_conj*P0).imag
            ek = (A + B*k**2)*(L*P1 + 1j*P2/lambda_0) + B * \
                lambda_0*(abs(lambda_0)**2)*P1 + 1j*B*lambda_0*P2
        res += ek
    return res


def optimal_alpha(g, omega, N):
    alpha_init = [10, -10]
    def e_reelle(alpha_parts): return abs(
        e(complex(alpha_parts[0], alpha_parts[1]), g, omega, N))
    res = optimize.minimize(e_reelle, alpha_init,
                            method='BFGS', options={'disp': DEBUG})
    return res.x


def compute_alpha(omega):
    alpha = optimal_alpha(g_3, omega, N)
    return alpha[0] - 1j*alpha[1]
    # return (5.5 + (15 - 5.5)*(1 - exp(-omega/5000))) - 1j*omega/300
