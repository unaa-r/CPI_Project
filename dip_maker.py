import numpy as np
from scipy.fft import fft, ifft
from scipy.special import erf
from multiprocessing import get_context
import time
import os
import multiprocessing as mp
import argparse

def Ereffer(Ec, Ea, w, tau):
    return (Ec + Ea) * np.exp(1j * w * tau)

#this phi is the dispersion from going through optical elements
def Esampler(Ec, Ea, phi):
    return (Ec - Ea) * np.exp(1j * phi)

# from https://www.coherent.com/resources/tech-notes/lasers/PropagationDispersionMeasurement_of_sub_10fsPulses_08_29_18.pdf
#Choose from "BK7", "Fused Silica", "Sapphire", "CaF2", or "SF10"
def glass_type_epsilon(w, w0, material="BK7"):
    
    c = 0.2998; #(*um/fs*)
    match material:
        case "BK7":
            b1 = 1.03961212 #(*for BK7 glass*)
            b2 = 0.231792344
            b3 = 1.01046945
            c1 = 6.00069867e-3 #(*um^2*)
            c2 = 2.00179144e-2
            c3 = 1.03560653e2
        case "Fused Silica":
            b1 =  0.6961663
            b2 = 0.4079426
            b3 = 0.8974794
            c1 = 0.00467914826
            c2 = 0.0135120631
            c3 = 97.9340025
        case "Sapphire":
            b1 = 1.43134930
            b2 = 0.650547130
            b3 = 5.34140210
            c1 = 0.00527992610
            c2 = 0.0142382647
            c3 = 325.017834
        case "CaF2":
            b1 = 0.5675888
            b2 = 0.4710914
            b3 = 3.8484723
            c1 = 0.00252642999
            c2 = 0.0100783328
            c3 = 1200.555973
        case "SF10":
            b1 = 1.62153902
            b2 = 0.256287842
            b3 = 1.64447552
            c1 = 0.0122241457
            c2 = 0.0595736775
            c3 = 103.560653
        case _:
            print("Error! invalid water type")
            return 0

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


#from https://opg.optica.org/ao/fulltext.cfm?uri=ao-36-16-3785
#T must be between 0 and 30
#S is salinity, in parts per thousand (35 is seawater)
def water_epsilon(w, w0, T=20, S=0):
    c = 0.2998; #(*um/fs*)
    
    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6

    lambdas = (2 * np.pi * c / w) * 1e3

    lambda_0 = (2*1000*np.pi*c/w0)

    k_w = w*(n0 + (n1 + n2*T + n3*T**2)*S + n4*T**2 +(n5 + n6*S + n7*T)/lambdas + 
             n8/lambdas**2 + n9/lambdas**3)/c
    k_deriv = (n0 + (n1 + n2*T + n3*T**2)*S + n4*T**2 +2*(n5 + n6*S + n7*T)/lambda_0 + 3*n8/lambda_0**2 + 4*n9/lambda_0**3)/c
    
    phi_water= k_w - k_deriv*(w - w0)

    return phi_water


def run_cpi_for_L(chirp_type, L, Ec, Ea, ws, taus, epsilon, integration_range, output_dir):
  
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--dispersion_type", type=str, default="BK7")
    parser.add_argument("--integration_range", type=int, default=1)
    parser.add_argument("--tau_range", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_L", type=int, default=64001)
    parser.add_argument("--L_stepsize", type=int, default=800)

    args = parser.parse_args()

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

    taus = np.arange(-args.tau_range, args.tau_range + .5, 0.5)
    folder_name = args.folder_name
    dispersion_type = args.dispersion_type #choose from seawater, freshwater, or BK7 glass for now (default BK7)
    integration_range = args.integration_range #units of nm, default 1 nm
    Lvals = np.arange(0, args.max_L, args.L_stepsize) #L values (thickness) in um, default use: np.arange(0, 64001, 800)
    overwrite = args.overwrite #choose to overwrite files in folder or not; false is good if run times out before you check all cases

    #################################

    os.makedirs(f"results/{folder_name}/linear", exist_ok=True)
    os.makedirs(f"results/{folder_name}/erf", exist_ok=True)
    os.makedirs(f"results/{folder_name}/super_erf", exist_ok=True)

    with open(f"results/{folder_name}/plotting_params.txt", 'w') as f:
        f.write(f"{dispersion_type}\n")
        f.write(f"{args.tau_range}\n")
        f.write(f"{args.max_L}\n")
        f.write(f"{args.stepsize}\n")

    with open(f"results/{folder_name}/run_info.txt", 'w') as f:
        f.write(f"Dispersion type: {dispersion_type}\n")
        f.write(f"Integration range: {integration_range} nm\n")
        f.write(f"Tau range: {args.tau_range} fs\n")
        f.write(f"Maximum L value (exclusive): {args.max_L} um\n")
        f.write(f"L stepsize: {args.stepsize} um\n")


    print(f"📊 Sweeping {len(Lvals)} dispersion values")

    # Build task list
    tasks = []

    ws[0] = 1e-6

    match dispersion_type:
        case "freshwater":
            epsilon = water_epsilon(ws, w_0, S = 0)
        case "seawater":    
            epsilon = water_epsilon(ws, w_0, S = 35)
        case "BK7":
            epsilon = glass_type_epsilon(ws, w_0)
        case "Fused Silica":
            epsilon = glass_type_epsilon(ws, w_0, "Fused Silica")
        case "Sapphire":
            epsilon = glass_type_epsilon(ws, w_0, "Sapphire")
        case "CaF2":
            epsilon = glass_type_epsilon(ws, w_0, "CaF2")
        case "SF10":
            epsilon = glass_type_epsilon(ws, w_0, "SF10")
        case _:
            epsilon = glass_type_epsilon(ws, w_0)        

    epsilon = np.nan_to_num(epsilon, nan=1e-6, posinf=1e-6, neginf=1e-6)
    epsilon = np.clip(epsilon, -1e6, 1e6)  # or tighter limits


    for L in Lvals:

        if not overwrite:
            if not (os.path.exists(f"./results/{folder_name}/linear/linear_L{L}.txt")):
                tasks.append(("linear", L, Ec_lin, Ea_lin, ws, taus, epsilon, integration_range, f"./results/{folder_name}/linear"))
            if not (os.path.exists(f"./results/{folder_name}/erf/erf_L{L}.txt")):
                tasks.append(("erf", L, Ec_erf, Ea_erf, ws, taus, epsilon, integration_range, f"./results/{folder_name}/erf"))
            if not (os.path.exists(f"./results/{folder_name}/super_erf/super_erf_L{L}.txt")):
                tasks.append(("super_erf", L, Ec_superf, Ea_superf, ws, taus, epsilon, integration_range, f"./results/{folder_name}/super_erf"))
        else:
            tasks.append(("linear", L, Ec_lin, Ea_lin, ws, taus, epsilon, integration_range, f"./results/{folder_name}/linear"))
            tasks.append(("erf", L, Ec_erf, Ea_erf, ws, taus, epsilon, integration_range, f"./results/{folder_name}/erf"))
            tasks.append(("super_erf", L, Ec_superf, Ea_superf, ws, taus, epsilon, integration_range, f"./results{folder_name}/super_erf"))
        
    # Run in parallel using 4 workers
    print("⚙️ Launching parallel CPI...")
    start = time.time()

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(run_cpi_for_L, tasks)

    print(f"\n✅ All CPI runs complete in {time.time() - start:.2f} s")