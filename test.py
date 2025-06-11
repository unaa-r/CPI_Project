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

def Esampler(Ec, Ea, w, w_0, phi):
    return (Ec - Ea) * np.exp(1j * phi)

def CPI(Ec, Ea, ws, w_0, taus, epsilon, queue, label):
    print(f"[{label}] Starting CPI...")
    Esamp_w = Esampler(Ec, Ea, ws, w_0, epsilon)
    Esamp_t = np.conj(ifft(np.conj(Esamp_w), norm="ortho"))

    SFG_data = []
    for tau in taus:
        Eref_w = Ereffer(Ec, Ea, ws, tau)
        Esfg_t = np.conj(ifft(np.conj(Eref_w), norm="ortho")) * Esamp_t
        Esfg_w = np.conj(fft(np.conj(Esfg_t), norm="ortho"))
        Isfg_w = np.abs(Esfg_w) ** 2
        SFG_data.append(Isfg_w)

    SFG_data = np.array(SFG_data)
    queue.put((label, SFG_data))
    print(f"[{label}] Finished CPI.")

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


def run_cpi_for_L(chirp_type, L, Ec, Ea, ws, w0, taus, integration_range, output_dir):
    
    phi = BK7_epsilon(ws, w0)*L
    phi = np.nan_to_num(phi, nan=1e-6, posinf=1e-6, neginf=1e-6)

    Esamp_w = Esampler(Ec, Ea, ws, w0, phi)
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

def erf_2(B, w, w_0, sigma):
    x = (w - w_0) * 2 * sigma / np.sqrt(2 * np.log(256))
    return B * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

def superf_chirp(C, w, w_0, sigma_s):
    x = (w - w_0) * 2 * sigma_s / np.sqrt(2 * np.log(256))
    return C * ((np.exp(-x**2) - 1) / np.sqrt(np.pi) + x * erf(x))

def chirper(Ews, phi):
    return Ews * np.exp(1j * phi)

# Constants
fwhm = 10
sigma_s = 1.12 * fwhm
c = 299792458
w_0 = 2 * np.pi * c * 1e-15 / 800e-9
t_0 = 200000
Npts = 2**19
ts = np.linspace(0, 400000, Npts)
dt = ts[1] - ts[0]
taus = np.arange(-50, 50.5, 0.5)

if __name__ == "__main__":
    start = time.time()
    ctx = get_context("spawn")  # safe for Windows/macOS

    # Constants
    fwhm = 10
    sigma_s = 1.12 * fwhm
    c = 299792458
    w_0 = 2 * np.pi * c * 1e-15 / 800e-9
    t_0 = 200000
    Npts = 2**19
    ts = np.linspace(0, 400000, Npts)
    dt = ts[1] - ts[0]
    taus = np.arange(-50, 50.5, 0.5)

    # Time-domain field
    Es = np.exp((-2 * np.log(2) * (ts - t_0) ** 2) / fwhm**2) * np.exp(-1j * w_0 * ts)
    Ews = np.conj(fft(np.conj(Es), norm='ortho'))
    ws = 2 * np.pi * np.arange(Npts) / (Npts * dt)

    # Chirps
    A, B, C = 180337, 8300, 7450

    Ec_lin = chirper(Ews, lin_chirp(A, ws, w_0))
    Ea_lin = chirper(Ews, lin_chirp(-A, ws, w_0))
    Ec_erf = chirper(Ews, erf_2(B, ws, w_0, fwhm))
    Ea_erf = chirper(Ews, erf_2(-B, ws, w_0, fwhm))
    Ec_superf = chirper(Ews, superf_chirp(C, ws, w_0, sigma_s))
    Ea_superf = chirper(Ews, superf_chirp(-C, ws, w_0, sigma_s))
    '''
    # Setup queues
    q1, q2, q3 = ctx.Queue(), ctx.Queue(), ctx.Queue()

    # Start processes
    p1 = ctx.Process(target=CPI, args=(Ec_lin, Ea_lin, ws, w_0, taus, 0, q1, "lin"))
    p2 = ctx.Process(target=CPI, args=(Ec_erf, Ea_erf, ws, w_0, taus, 0, q2, "erf"))
    p3 = ctx.Process(target=CPI, args=(Ec_superf, Ea_superf, ws, w_0, taus, 0, q3, "superf"))

    # Start processes
    p1.start(); p2.start(); p3.start()

    # ⚠️ Get results BEFORE join to avoid deadlock
    label1, lin_cpi = q1.get()
    label2, erf_cpi = q2.get()
    label3, superf_cpi = q3.get()

    # Now safe to join
    p1.join(); p2.join(); p3.join()

    print("✅ All processes joined.")


    print(f"\n✅ Done in {time.time() - start:.2f} seconds")

    #skipping the first bin of everything now
    ws[0] = 1e-6

    # --- Generate wavelength axis ---
    wavelengths = 2 * np.pi * c * 1e9 / (ws * 1e15)

    # --- Correct indices (Python is 0-based) ---
    start = 299753
    end = 299833

    # --- Normalize datasets ---
    lin_cpi_norm = lin_cpi / np.max(lin_cpi)
    erf_cpi_norm = erf_cpi / np.max(erf_cpi)
    superf_cpi_norm = superf_cpi / np.max(superf_cpi)

    # --- Compute plotting extent ---
    extent = [taus[0], taus[-1], wavelengths[end], wavelengths[start]]

    # --- Set up figure with GridSpec (3 plots + 1 for colorbar) ---
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    datasets = [lin_cpi_norm, erf_cpi_norm, superf_cpi_norm]
    titles = ['Linear CPI', 'ERF CPI', 'Super ERF CPI']

    # --- Plot heatmaps ---
    for ax, data, title in zip(axes, datasets, titles):
        im = ax.imshow(
            data[:, start:end].T,
            extent=extent,
            origin='lower',
            aspect='auto',
            cmap='inferno'
        )
        ax.set_xlabel('Time Delay τ (fs)')
        ax.set_title(f'SFG Intensity - {title}')
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    axes[0].set_ylabel('Wavelength λ (nm)')

    # --- Add colorbar in its own subplot space ---
    cbar_ax = fig.add_subplot(gs[3])
    fig.colorbar(im, cax=cbar_ax, label='SFG Intensity (a.u.)')

    plt.tight_layout()
    plt.show()
    '''
    signals = []
    curve_fits = []
    params = []

    def one_minus_gaussian(t, t_0, a1, a2, fwhm):
        return a1 - a2*np.exp(-4*math.log(2)*((t - t_0)**2)/fwhm**2)

    def parabolic_one_minus_gaussian(t, t_0, a1, a2, a3, fwhm):
        return (a1 - a2*(t - t_0)**2)*(1 - a3*np.exp(-4*math.log(2)*((t - t_0)**2)/fwhm**2))

    def chisq(params, t, y):
        t0, a1, a2, fwhm = params
        model = one_minus_gaussian(t, t0, a1, a2, fwhm)
        return np.sum((y - model)**2)

    def chisq_parabolic(params, t, y):
        t0, a1, a2, a3, fwhm = params
        model = parabolic_one_minus_gaussian(t, t0, a1, a2, a3, fwhm)
        return np.sum((y - model)**2)
    '''
    fit_names= []

    for d, name in zip(datasets, titles):
        SFG_band = d[:, (wavelengths >= 399.5) & (wavelengths <= 400.5)]
        signal_vs_tau = np.sum(SFG_band, axis=1)  # integrate over wavelength axis

        signal_vs_tau_norm = (signal_vs_tau - np.min(signal_vs_tau))/(np.max(signal_vs_tau) - np.min(signal_vs_tau))
        signals.append(signal_vs_tau_norm)

        # --- Initial guesses ---
        init_guess = [0.0, 0.0025, 0.9, 10.0]  # Corrected to 3 parameters

        # --- Fit the data ---
        result = minimize(chisq, init_guess, args=(taus, signal_vs_tau_norm), method='L-BFGS-B')

        # Extract the parameters
        params.append(result.x)

        # Calculate the final chi-squared values
        final_chisq = chisq(result.x, taus, signal_vs_tau_norm)

        # Print the fit parameters and chi-squared values
        print(f"Fit Parameters (t0, a1, a2, fwhm) for {name}: {result.x}")
        print(f"Final Chi-Squared Value for {name}: {final_chisq}")

        # Evaluate the fits
        curve_fits.append(one_minus_gaussian(taus, *result.x))
        fit_names.append("1 - Gaussian")

    #parabolic
    for d, name in zip(datasets, titles):
        SFG_band = d[:, (wavelengths >= 399.5) & (wavelengths <= 400.5)]
        signal_vs_tau = np.sum(SFG_band, axis=1)  # integrate over wavelength axis
        signal_vs_tau_norm = (signal_vs_tau - np.min(signal_vs_tau))/(np.max(signal_vs_tau) - np.min(signal_vs_tau))
        signals.append(signal_vs_tau_norm)

        # --- Initial guesses ---
        init_guess_parabolic = [0, 0.0025, 0.00000001, 0.9, 10]

        # --- Fit the data ---
        result2 = minimize(chisq_parabolic, init_guess_parabolic, args=(taus, signal_vs_tau_norm), method='L-BFGS-B')

        # Extract the parameters
        params.append(result2.x)

        # Calculate the final chi-squared values
        final_chisq2 = chisq_parabolic(result2.x, taus, signal_vs_tau_norm)

        # Print the fit parameters and chi-squared values

        print(f"Fit Parameters (t0, a1, a2, a3, fwhm) for {name} (parabolic): {result2.x}")
        print(f"Final Chi-Squared Value for {name} (parabolic): {final_chisq2}")

        # Evaluate the fits
        curve_fits.append(parabolic_one_minus_gaussian(taus, *result2.x))

        fit_names.append("Parabolic 1 - Gaussian")

        complete_titles = ["Linear CPI", "ERF CPI", "Super ERF CPI", "Linear CPI - Parabolic", "ERF CPI - Parabolic", "Super ERF CPI - Parbolic"]
        # --- Create subplots ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 5), sharey=True)

        axes = axes.flatten()

        for ax, signal, curve, name, fit in zip(axes, signals, curve_fits, complete_titles, fit_names):
            ax.plot(taus, signal, 'o', label='Normalized Data', color='#4B25B1')
            ax.plot(taus, curve, '-', label=fit, color='black')
            ax.set_xlabel('Time Delay τ (fs)')
            ax.set_title(name)
            ax.legend()
            ax.grid(True)

            # --- Optional: print fitted parameters ---
            #print(f"Fitted parameters for {name}: a = {param[0]:.4f}, t0 = {param[1]:.4f} fs, FWHM = {param[2]:.4f} fs")
            
        axes[0].set_ylabel('Normalized Intensity')
        plt.tight_layout()
        plt.show();
        '''
    os.makedirs("updated_phi_results/linear", exist_ok=True)
    os.makedirs("updated_phi_results/erf", exist_ok=True)
    os.makedirs("updated_phi_results/super_erf", exist_ok=True)

    # L values
    Lvals = np.arange(0, 64001, 800)
    Lvals2 = [0, 800]
    print(f"📊 Sweeping {len(Lvals)} dispersion values")

    # Build task list
    tasks = []

    ws[0] = 1e-6
    
    for L in Lvals:

        tasks.append(("linear", L, Ec_lin, Ea_lin, ws, w_0, taus, 1, "./updated_phi_results/linear"))
        tasks.append(("erf", L, Ec_erf, Ea_erf, ws, w_0, taus, 1, "./updated_phi_results/erf"))
        tasks.append(("super_erf", L, Ec_superf, Ea_superf, ws, w_0, taus, 1, "./updated_phi_results/super_erf"))

    # Run in parallel using 4 workers
    print("⚙️ Launching parallel CPI...")
    start = time.time()

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(run_cpi_for_L, tasks)

    print(f"\n✅ All CPI runs complete in {time.time() - start:.2f} s")

