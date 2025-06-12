import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, invertlaplace
from scipy.special import jn_zeros

#=======================================
# 1) Paramètres physiques et domaines
#=======================================
# Fluides
rho1, rho2 = 1.3, 997.0           # kg/m³
mu1,   mu2   = 0.018, 0.1         # Pa·s
gamma, g    = 0.072, 9.81         # N/m, m/s²
rho_sum     = rho1 + rho2
rho_diff    = rho2 - rho1
mu_sum      = mu1 + mu2

# Domaine spatial unique (pour toutes les interfaces)
x = np.linspace(0, 2*np.pi, 1000)

# Domaine en k unique (pour toutes les dispersions)
k_space = np.linspace(0.1, 600, 1000)

# Onde de référence pour tracer η(x,t)
k_val = 1.0

# Amplitude initiale
a0 = 5e-4

#=======================================
# 2) Fonctions non-visqueuses
#=======================================
def omega_nv_squared(k):
    # w0^2 = (Δρ g k – γ k^3) / (ρ1+ρ2)
    return (rho_diff*g*k - gamma*k**3)/rho_sum

def omega_nv_signed(k):
    w2 = omega_nv_squared(k)
    return np.sign(w2) * np.sqrt(np.abs(w2))

def eta_nv(k, t, a0=a0):
    w2 = omega_nv_squared(k)
    if w2 > 0:
        ω = np.sqrt(w2)
        at = a0 * np.sinh(ω*t)
    elif w2 < 0:
        ω = np.sqrt(-w2)
        at = a0 * np.sin(ω*t)
    else:
        at = a0 * t
    return at * np.cos(k * x)

#=======================================
# 3) Fonctions viscosité simplifiée
#=======================================
def beta_v(k):
    return 2*mu_sum * k**2 / rho_sum

def omega_v_squared(k):
    # on reprend l'expression négative de w0^2 pour l'instable
    return (gamma*k**3 - rho_diff*g*k)/rho_sum

def delta_v(k):
    return beta_v(k)**2 - 4 * omega_v_squared(k)

def sigma_v(k):
    b = beta_v(k)
    d = delta_v(k)
    # pour d>0 deux racines réelles, on prend la plus grande
    return np.where(d>0, (-b + np.sqrt(d))/2, -b/2)

def eta_v_simpl(k, t, a0=a0):
    b = beta_v(k)
    d = delta_v(k)
    D = np.sqrt(np.abs(d))/2
    expf = np.exp(-b*t/2)
    # si d<0 -> régime oscillant amorti
    osc = np.cos(D*t) + (b/(2*D))*np.sin(D*t)
    # si d>0 -> croissance/decroissance hyperb. amortie
    hyp = np.cosh(D*t) + (b/(2*D))*np.sinh(D*t)
    at = a0 * expf * np.where(d<0, osc, hyp)
    return at * np.cos(k * x)

#=======================================
# 4) Inversion de Laplace (modèle complet)
#=======================================
# On reprend vos définitions de beta(s), atilde(s)
# –––––––––––––––––––––––––––––––––––––––––––––––––––
# Exemple générique (à adapter si vos expressions diffèrent)
# Notons ici :
#   nu1 = mu1/rho1, nu0 = mu0/rho0, R, k_laplace, omega0 déjà définis
#–––––––––––––––––––––––––––––––––––––––––––––––––––––
# Pour l’exemple, on choisit le même mode k_val pour le laplace
k_laplace = k_val
R = 2*np.pi / k_val  # periodique sur [0,2π] → longueur d’onde λ=2π/k → R=λ
nu1 = mu1 / rho1
nu0 = mu2 / rho2

# Exemples de lambda(s) et beta(s) issues de votre travail :
lam1 = lambda s: s/nu1 + k_laplace**2
lam0 = lambda s: s/nu0 + k_laplace**2

zeta = lambda s: 2*(k_laplace + lam1(s))*(mu1*k_laplace + mu2*lam0(s)) / \
                (mu2*(k_laplace + lam0(s)) + mu1*(k_laplace + lam1(s)))
xi   = lambda s: 2*(k_laplace + lam0(s))*(mu2*k_laplace + mu1*lam1(s)) / \
                (mu2*(k_laplace + lam0(s)) + mu1*(k_laplace + lam1(s)))

beta_s = lambda s: (2*k_laplace**2*(mu1+mu2)
                    + k_laplace*(mu1*zeta(s) + mu2*xi(s))
                    - 2*k_laplace**2*((mu1*zeta(s))/(lam1(s)+k_laplace)
                                     + (mu2*xi(s))/(lam0(s)+k_laplace))
                   ) / rho_sum

atilde = lambda s: ((s + beta_s(s))*a0) / (s**2 + beta_s(s)*s + omega_nv_squared(k_laplace))

mp.dps = 20  # précision
def a_of_t(t):
    if t == 0:
        return a0
    return float(invertlaplace(atilde, t, method='stehfest'))

def eta_v_laplace(t):
    return a_of_t(t) * np.cos(k_laplace * x)

#=======================================
# 5) Routine de dichotomie (pour kc visqueux simplifié)
#=======================================
def dichotomie(f, a, b, tol=1e-6, maxiter=10000):
    fa, fb = f(a), f(b)
    if fa*fb >= 0:
        raise ValueError("Changement de signe requis en [a,b]")
    for _ in range(maxiter):
        c = 0.5*(a + b)
        fc = f(c)
        if abs(fc) < tol or (b-a)/2 < tol:
            return c
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5*(a+b)

#=======================================
# 6) Tracés
#=======================================
times = np.arange(0, 0.6, 0.1)

#--- Non-visqueux ---
kc_nv = np.sqrt(g*rho_diff/gamma)
omega_vals = omega_nv_signed(k_space)
k0_nv = k_space[np.argmax(omega_vals)]

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
for t in times:
    ax1.plot(x, eta_nv(k_val, t), label=f"t={t:.1f}s")
ax1.set(title="Interface non-visqueuse", xlabel="x", ylabel="η(x,t)")
ax1.legend(); ax1.grid()

ax2.plot(k_space, omega_vals, label=r"$\omega(k)$")
ax2.axhline(0, color='k', ls='--')
ax2.axvline(kc_nv, color='gray', ls=':')
ax2.fill_between(k_space, omega_vals, 0,
                 where=omega_vals>=0, color='salmon', alpha=0.3, label="Instable")
ax2.fill_between(k_space, omega_vals, 0,
                 where=omega_vals<0, color='skyblue',alpha=0.3, label="Stable")
ax2.scatter([kc_nv, k0_nv],[0, omega_vals.max()])
ax2.text(kc_nv, 0, r'$k_c$', va='bottom')
ax2.text(k0_nv, omega_vals.max(), r'$k_0$', va='top')
ax2.set(title="Dispersion non-visqueuse", xlabel="k", ylabel=r"$\omega(k)$")
ax2.legend(); ax2.grid()
plt.tight_layout()
plt.show()


#--- Visc. simplifiée ---
kc_v = dichotomie(sigma_v, 300, 400)
sigma_vals = sigma_v(k_space)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
for t in times:
    ax1.plot(x, eta_v_simpl(k_val, t), label=f"t={t:.1f}s")
ax1.set(title="Interface visqueuse simplifiée", xlabel="x", ylabel="η(x,t)")
ax1.legend(); ax1.grid()

ax2.plot(k_space, sigma_vals, label=r"$\sigma(k)$")
ax2.axhline(0, color='k', ls='--')
ax2.axvline(kc_v, color='gray', ls=':')
ax2.fill_between(k_space, sigma_vals, 0,
                 where=sigma_vals>=0, color='salmon', alpha=0.3, label="Instable")
ax2.fill_between(k_space, sigma_vals, 0,
                 where=sigma_vals<0, color='skyblue', alpha=0.3, label="Stable")
ax2.scatter([kc_v, k_space[np.argmax(sigma_vals)]],
            [0, sigma_vals.max()])
ax2.text(kc_v, 0, r'$k_c$', va='bottom')
ax2.text(k_space[np.argmax(sigma_vals)], sigma_vals.max(), r'$k_0$', va='top')
ax2.set(title="Dispersion visqueuse simplifiée", xlabel="k", ylabel=r"$\sigma(k)$")
ax2.legend(); ax2.grid()
plt.tight_layout()
plt.show()


#--- Modèle complet (Laplace) ---
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))
for t in times:
    ax1.plot(x, eta_v_laplace(t), label=f"t={t:.1f}s")
ax1.set(title="Interface visqueuse (modèle complet)", xlabel="x", ylabel="η(x,t)")
ax1.legend(); ax1.grid()

# On réaffiche la dispersion simplifiée en face
for t in times:
    ax2.plot(x, eta_v_simpl(k_val, t), label=f"t={t:.1f}s")
ax2.set(title="Interface visqueuse simplifiée", xlabel="x", ylabel="η(x,t)")
ax2.legend(); ax2.grid()
plt.tight_layout()
plt.show()