import numpy as np
from scipy.fft import fft, ifft
from scipy.special import erf
from multiprocessing import get_context
import time
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize 
import math
import os
import multiprocessing as mp

def Ereffer(Ec, Ea, w, tau):
    return (Ec + Ea) * np.exp(1j * w * tau)

#this phi is the dispersion from going through optical elements
def Esampler(Ec, Ea, phi):
    return (Ec - Ea) * np.exp(1j * phi)

def BK7_epsilon(w, w0):
    c = 0.2998; #(*um/fs*)
    b1 = 1.03961212; #(*for BK7 glass*)
    b2 = 0.231792344
    b3 = 1.01046945
    c1 = 6.00069867e-3 #(*um^2*)
    c2 = 2.00179144e-2
    c3 = 1.03560653e2

    k_w = w*(np.sqrt(1 + (b1*(2*np.pi*c/w)**2)/((2*np.pi*c/w)**2 - c1) + 
                     (b2*(2*np.pi*c/w)**2)/((2*np.pi*c/w)**2 - c2) + 
                     (b3*(2*np.pi*c/w)**2)/((2*np.pi*c/w)**2 - c3)))/c
    
    k_deriv = np.sqrt(1 +((4*b1*(c**2)*(np.pi**2))/(-c1*(w0**2) + (2*c*np.pi)**2)) + 
                      ((4*b2*(c**2)*(np.pi**2))/(-c2*(w0**2) + (2*c*np.pi)**2)) + 
                      ((4*b3*(c**2)*(np.pi**2))/(-c3*(w0**2) + (2*c*np.pi)**2)))/c + w0*(
                          (32*b1*(c*np.pi)**4)/((w0**5)*((-c1 + ((2*c*np.pi)**2)/(w0**2))**2)) +
                          (32*b2*(c*np.pi)**4)/((w0**5)*((-c2 + ((2*c*np.pi)**2)/(w0**2))**2)) +
                          (32*b3*(c*np.pi)**4)/((w0**5)*((-c3 + ((2*c*np.pi)**2)/(w0**2))**2)) -
                          (8*b1*(c*np.pi)**2)/((w0**3)*(-c1 + ((2*c*np.pi)**2)/(w0**2))) -
                          (8*b2*(c*np.pi)**2)/((w0**3)*(-c2 + ((2*c*np.pi)**2)/(w0**2))) -
                          (8*b3*(c*np.pi)**2)/((w0**3)*(-c3 + ((2*c*np.pi)**2)/(w0**2))))/(
                              2*c*np.sqrt(1 + (b1*(2*np.pi*c/w0)**2)/((2*np.pi*c/w0)**2 - c1) + 
                                          (b2*(2*np.pi*c/w0)**2)/((2*np.pi*c/w0)**2 - c2) + 
                                          (b3*(2*np.pi*c/w0)**2)/((2*np.pi*c/w0)**2 - c3)))

    return k_w - k_deriv*(w - w0)

#from https://research.engr.oregonstate.edu/parrish/index-refraction-seawater-and-freshwater-function-wavelength-and-temperature#:~:text=Christopher%20Parrish%20(2020),400%2D700%20=%20visible%20spectrum)
#T must be between 0 and 30
#can pick from 'fresh' (salinity = 0), or 'sea' (salinity = 35%)
def water_epsilon(w, w0, T=20, water='fresh'):
    c = 0.2998; #(*um/fs*)
    if(water == 'fresh'):
        a = -1.97812e-6
        b = 1.03223e-7
        c1 = -8.58125e-6
        d = -1.54834e-4
        e = 1.38919
    elif(water == 'sea'):
        a = -1.50156e-6
        b = 1.07085e-7
        c1 = -4.27594e-5
        d = -1.60476e-4
        e = 1.39807
    else:
        print("Error! invalid water type")
        return 0

    k_w = w*(a*T**2 + b*(2*1000*np.pi*c/w)**2 + c1*T + d*2*1000*np.pi*c/w + e)/c
    k_deriv = a - b/(w**2) + c1 + e

    return k_w - k_deriv*(w - w0)


def run_cpi_for_L(chirp_type, L, Ec, Ea, ws, w0, taus, epsilon, integration_range, output_dir):
  
    Esamp_w = Esampler(Ec, Ea, epsilon*L)
    Esamp_t = np.conj(ifft(np.conj(Esamp_w), norm="ortho"))

    SFG_data = []
    for tau in taus:
        Eref_w = Ereffer(Ec, Ea, ws, tau)
        Esfg_t = np.conj(ifft(np.conj(Eref_w), norm="ortho")) * Esamp_t
        Esfg_w = np.conj(fft(np.conj(Esfg_t), norm="ortho"))
        Isfg_w = np.abs(Esfg_w)**2
        SFG_data.append(Isfg_w)

    SFG_data = np.array(SFG_data)

    wavelengths = 2 * np.pi * c * 1e9 / (ws * 1e15)

    SFG_band = SFG_data[:, (wavelengths >= 400 - (integration_range/2)) & (wavelengths <= 400 + (integration_range/2))]
    signal_vs_tau = np.sum(SFG_band, axis=1)
    
   
    out_path = os.path.join(output_dir, f"{chirp_type}_L{L}.txt")
    np.savetxt(out_path, signal_vs_tau, fmt="%.15f")
    print(f"✅ {chirp_type} L={L} saved.")


def lin_chirp(A, w, w_0):
    return A * ((w - w_0) ** 2)

def erf_chirp(B, w, w_0, sigma):
    x = (w - w_0) * 2 * sigma / np.sqrt(2 * np.log(256))
    return B * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

def superf_chirp(C, w, w_0, sigma_s):
    x = (w - w_0) * 2 * sigma_s / np.sqrt(2 * np.log(256))
    return C * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

#this phi is the chirp you want to apply to the pulse
def chirper(Ews, phi):
    return Ews * np.exp(1j * phi)

# Constants
fwhm = 10 #fs
sigma_s = 1.12 * fwhm
c = 299792458 #m/s
w_0 = 2 * np.pi * c * 1e-15 / 800e-9 #fs^-1
t_0 = 200000 #fs
Npts = 2**19
ts = np.linspace(0, 400000, Npts)
dt = ts[1] - ts[0]


if __name__ == "__main__":
    start = time.time()
    ctx = get_context("spawn")  # safe for Windows/macOS

    # Time-domain field
    Es = np.exp((-2 * np.log(2) * (ts - t_0) ** 2) / fwhm**2) * np.exp(-1j * w_0 * ts)
    Ews = np.conj(fft(np.conj(Es), norm='ortho'))
    ws = 2 * np.pi * np.arange(Npts) / (Npts * dt)

    # Chirps
    A, B, C = 180337, 8300, 7450

    Ec_lin = chirper(Ews, lin_chirp(A, ws, w_0))
    Ea_lin = chirper(Ews, lin_chirp(-A, ws, w_0))
    Ec_erf = chirper(Ews, erf_chirp(B, ws, w_0, fwhm))
    Ea_erf = chirper(Ews, erf_chirp(-B, ws, w_0, fwhm))
    Ec_superf = chirper(Ews, superf_chirp(C, ws, w_0, sigma_s))
    Ea_superf = chirper(Ews, superf_chirp(-C, ws, w_0, sigma_s))

    ###This is what you can modify###
    taus = np.arange(-10, 10.5, 0.5)
    folder_name = "freshwater_dispersion"
    dispersion_type = "freshwater" #choose from seawater, freshwater, or BK7 glass for now (default BK7)
    integration_range = 1 #units of nm, default 1 nm
    Lvals = np.arange(0, 64001, 800) #L values (thickness) in um, default use: np.arange(0, 64001, 800)
    #################################

    os.makedirs("{folder_name}/linear", exist_ok=True)
    os.makedirs("{folder_name}/erf", exist_ok=True)
    os.makedirs("{folder_name}/super_erf", exist_ok=True)

    print(f"📊 Sweeping {len(Lvals)} dispersion values")

    # Build task list
    tasks = []

    ws[0] = 1e-6

    if(dispersion_type == "freshwater"):
        epsilon = water_epsilon(ws, w_0, water = 'fresh')
        epsilon = np.nan_to_num(epsilon, nan=1e-6, posinf=1e-6, neginf=1e-6)
    elif(dispersion_type == "seawater"):
        epsilon = water_epsilon(ws, w_0, water = 'sea')
        epsilon = np.nan_to_num(epsilon, nan=1e-6, posinf=1e-6, neginf=1e-6)
    elif(dispersion_type == "BK7"):
        epsilon = BK7_epsilon(ws, w_0)
        epsilon = np.nan_to_num(epsilon, nan=1e-6, posinf=1e-6, neginf=1e-6)
    else:
        epsilon = BK7_epsilon(ws, w_0)
        epsilon = np.nan_to_num(epsilon, nan=1e-6, posinf=1e-6, neginf=1e-6)        

    for L in Lvals:
        tasks.append(("linear", L, Ec_lin, Ea_lin, ws, w_0, taus, epsilon, integration_range, "./{folder_name}/linear"))
        tasks.append(("erf", L, Ec_erf, Ea_erf, ws, w_0, taus, epsilon, integration_range, "./{folder_name}/erf"))
        tasks.append(("super_erf", L, Ec_superf, Ea_superf, ws, w_0, taus, epsilon, integration_range, "./{folder_name}/super_erf"))

    # Run in parallel using 4 workers
    print("⚙️ Launching parallel CPI...")
    start = time.time()

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(run_cpi_for_L, tasks)

    print(f"\n✅ All CPI runs complete in {time.time() - start:.2f} s")