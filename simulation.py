import numpy as np
import matplotlib.pyplot as plt

def Conserved(rho, vx, vy, p, gamma, volume):

    mass = rho * volume
    momx = vx * mass
    momy = vy * mass
    energy = (p / (gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)) * volume

    return mass, momx, momy, energy

def Primitive(mass, momx, momy, energy, gamma, volume):

    rho = mass / volume
    vx = momx / rho / volume
    vy = momy / rho / volume
    p = (energy / volume - 0.5 * rho * (vx ** 2 + vy ** 2)) * (gamma - 1)

    return rho, vx, vy, p

def Gradient(f, dx):

    f_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    f_dy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)

    return f_dx, f_dy

def SlopeLimiter(f, dx, f_dx, f_dy):

    f_dx = np.maximum(0, np.minimum(1, ((f - np.roll(f, 1, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx
    f_dx = np.maximum(0, np.minimum(1, (-(f - np.roll(f, -1, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx
    f_dy = np.maximum(0, np.minimum(1, ((f - np.roll(f, 1, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)))) * f_dy
    f_dy = np.maximum(0, np.minimum(1, (-(f - np.roll(f, -1, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)))) * f_dy

    return f_dx, f_dy

def FaceExtrapolation(f, f_dx, f_dy, dx):

    f_XL = f - f_dx * dx / 2
    f_XL = np.roll(f_XL, -1, axis=0)
    f_XR = f + f_dx * dx / 2

    f_YL = f - f_dy * dx / 2
    f_YL = np.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy * dx / 2

    return f_XL, f_XR, f_YL, f_YR

def GetFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, p_L, p_R, gamma):

    en_L = p_L / (gamma - 1) + 0.5 * rho_L * (vx_L ** 2 + vy_L ** 2)
    en_R = p_R / (gamma - 1) + 0.5 * rho_R * (vx_R ** 2 + vy_R ** 2)

    # Stany uśrednione
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)
    p_star = (gamma - 1) * (en_star - 0.5 * (momx_star ** 2 + momy_star ** 2) / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star ** 2 / rho_star + p_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Energy = (en_star + p_star) * momx_star / rho_star

    # Wave speed
    c_L = np.sqrt(gamma * p_L / rho_L) + np.abs(vx_L)
    c_R = np.sqrt(gamma * p_R / rho_R) + np.abs(vx_R)
    c = np.maximum(c_L, c_R)

    flux_Mass -= c * 0.5 * (rho_L - rho_R)
    flux_Momx -= c * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= c * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= c * 0.5 * (en_L - en_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def ApplyFluxes(f, flux_F_X, flux_F_Y, dx, dt):

    f += - dt * dx * flux_F_X
    f += dt * dx * np.roll(flux_F_X, 1, axis=0)
    f += - dt * dx * flux_F_Y
    f += dt * dx * np.roll(flux_F_Y, 1, axis=1)

    return f

def main():
    ################# SIMULATION PARAMETERS ###################

    gamma = 5 / 3
    cfl = 0.4
    N = 512
    boxsize = 1.0
    t = 0
    t_end = 2
    dx = boxsize / N
    vol = dx ** 2
    xlin = np.linspace(0.5 * dx, boxsize - dx * 0.5, N)
    Y, X = np.meshgrid(xlin, xlin)

    ############## PERTURBATION #########################

    w0 = 0.1
    sigma = 0.05 / np.sqrt(2.)
    rho = 1. + (np.abs(Y - 0.5) < 0.25)
    vx = -0.5 + (np.abs(Y - 0.5) < 0.25)
    vy = w0 * np.sin(4 * np.pi * X) * (
                    np.exp(-(Y - 0.25) ** 2 / (2 * sigma ** 2)) + np.exp(-(Y - 0.75) ** 2 / (2 * sigma ** 2)))
    p = 2.5 * np.ones(X.shape)

    mass, momx, momy, energy = Conserved(rho, vx, vy, p, gamma, vol)

    fig = plt.figure(figsize=(8, 8), dpi=80)

    while t < t_end:

        rho, vx, vy, p = Primitive(mass, momx, momy, energy, gamma, vol)

        dt = cfl * np.min(dx / (np.sqrt(gamma * p / rho) + np.sqrt(vx ** 2 + vy ** 2)))

        rho_dx, rho_dy = Gradient(rho, dx)
        vx_dx, vx_dy = Gradient(vx, dx)
        vy_dx, vy_dy = Gradient(vy, dx)
        p_dx, p_dy = Gradient(p, dx)

        rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * p_dx)
        vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * p_dy)
        p_prime = p - 0.5 * dt * (gamma * p * (vx_dx + vy_dy) + vx * p_dx + vy * p_dy)

        rho_XL, rho_XR, rho_YL, rho_YR = FaceExtrapolation(rho_prime, rho_dx, rho_dy, dx)
        vx_XL, vx_XR, vx_YL, vx_YR = FaceExtrapolation(vx_prime, vx_dx, vx_dy, dx)
        vy_XL, vy_XR, vy_YL, vy_YR = FaceExtrapolation(vy_prime, vy_dx, vy_dy, dx)
        p_XL, p_XR, p_YL, p_YR = FaceExtrapolation(p_prime, p_dx, p_dy, dx)

        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = GetFlux(rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, p_XL, p_XR, gamma)
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = GetFlux(rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, p_YL,
                                                                       p_YR, gamma)

        mass = ApplyFluxes(mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        momx = ApplyFluxes(momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        momy = ApplyFluxes(momy, flux_Momy_X, flux_Momy_Y, dx, dt)
        energy = ApplyFluxes(energy, flux_Energy_X, flux_Energy_Y, dx, dt)

        t += dt

        if t <= t_end:
            plt.cla()
            plt.imshow(rho)
            plt.clim(0.8, 2.2)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)

#    plt.savefig('finitevolume.png', dpi=240)
    plt.show()

    return 0

if __name__ == "__main__":
    main()



