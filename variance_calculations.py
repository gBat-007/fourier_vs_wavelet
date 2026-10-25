import numpy as np
from scipy import signal, fft
import pywt

fs = 400.0
dt = 1.0 / fs

t_basis = np.linspace(-10, 10, 8000)
t_pulse = np.linspace(-5, 5, 4000)
f1 = np.exp(-(t_pulse**2)/2)

T_chirp = 10.0
t_chirp = np.arange(0.0, T_chirp, dt)
f2 = np.cos(t_chirp**2)

alphas = [0.10, 0.25, 0.50]
scales_test = [0.5, 1.0, 2.0]  # For Stage 1
wavelet = "cmor1.5-1.0"

cwt_scale_configs = [
    {"name": "narrow", "center_freq": 20, "bandwidth": 0.3},  # High freq, narrow time
    {"name": "medium", "center_freq": 8, "bandwidth": 0.3},   # Medium
    {"name": "wide", "center_freq": 2, "bandwidth": 0.3},     # Low freq, wide time
]

def compute_1d_uncertainty(g, t):
    dt_val = np.mean(np.diff(t))
    
    rho_t = np.abs(g)**2
    E = np.sum(rho_t) * dt_val
    if E < 1e-12:
        return np.nan, np.nan, np.nan
    rho_t = rho_t / E
    
    mu_t = np.sum(t * rho_t) * dt_val
    sigma_t2 = np.sum((t - mu_t)**2 * rho_t) * dt_val
    sigma_t = np.sqrt(sigma_t2)
    
    G = fft.fftshift(fft.fft(fft.ifftshift(g)))
    N = len(g)
    omega = 2 * np.pi * fft.fftshift(fft.fftfreq(N, dt_val))
    
    rho_omega = np.abs(G)**2
    domega = np.mean(np.diff(omega))
    E_omega = np.sum(rho_omega) * domega / (2*np.pi)
    rho_omega = rho_omega / E_omega / (2*np.pi)
    
    mu_omega = np.sum(omega * rho_omega) * domega
    sigma_omega2 = np.sum((omega - mu_omega)**2 * rho_omega) * domega
    sigma_omega = np.sqrt(sigma_omega2)
    
    U = sigma_t * sigma_omega
    return sigma_t, sigma_omega, U

# ============================
# STAGE 1
# ============================
print("\n" + "="*60)
print("STAGE 1: Basis Function Uncertainty")
print("="*60 + "\n")

print("STFT Gaussian Window")
print("-" * 60)
for alpha in alphas:
    C = (np.pi * alpha**2)**(-0.25)
    h = C * np.exp(-t_basis**2 / (2 * alpha**2))
    sigma_t, sigma_omega, U = compute_1d_uncertainty(h, t_basis)
    print(f"α = {alpha:.2f} s: σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")

print("\nMorlet Wavelet at different scales")
print("-" * 60)
cwt_obj = pywt.ContinuousWavelet(wavelet)
psi_base, t_base = cwt_obj.wavefun(level=10)
t_base_scaled = t_base * 2.0

for a in scales_test:
    t_scaled = t_basis / a
    psi_real = np.interp(t_scaled, t_base_scaled, np.real(psi_base), left=0, right=0)
    psi_imag = np.interp(t_scaled, t_base_scaled, np.imag(psi_base), left=0, right=0)
    psi_a = (psi_real + 1j * psi_imag) / np.sqrt(np.abs(a))
    sigma_t, sigma_omega, U = compute_1d_uncertainty(psi_a, t_basis)
    print(f"Scale a = {a:.1f}: σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")

# ============================
# STAGE 2A
# ============================
print("\n" + "="*60)
print("STAGE 2A: Signal Intrinsic Uncertainty")
print("="*60 + "\n")

print("Gaussian pulse f₁(t) = exp(-t²/2)")
sigma_t, sigma_omega, U = compute_1d_uncertainty(f1, t_pulse)
print(f"  σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")

print("\nChirp f₂(t) = cos(t²)")
sigma_t, sigma_omega, U = compute_1d_uncertainty(f2, t_chirp)
print(f"  σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")

# ============================
# 2D Concentration
# ============================
def compute_2d_tf_concentration(P, t_axis, f_axis):
    dt_bin = np.mean(np.diff(t_axis)) if len(t_axis) > 1 else 1.0
    df_bin = np.mean(np.diff(f_axis)) if len(f_axis) > 1 else 1.0
    
    total_E = np.sum(P) * dt_bin * df_bin
    if total_E == 0:
        return np.nan, np.nan, np.nan
    P_norm = P / total_E
    
    mu_t = np.sum(P_norm * t_axis[None, :]) * dt_bin * df_bin
    mu_f = np.sum(P_norm * f_axis[:, None]) * dt_bin * df_bin
    
    sigma_t2 = np.sum(P_norm * (t_axis[None, :] - mu_t)**2) * dt_bin * df_bin
    sigma_f2 = np.sum(P_norm * (f_axis[:, None] - mu_f)**2) * dt_bin * df_bin
    
    sigma_t = np.sqrt(sigma_t2)
    sigma_omega = 2 * np.pi * np.sqrt(sigma_f2)
    return sigma_t, sigma_omega, sigma_t * sigma_omega

def gaussian_window(alpha_seconds, fs, n_std=4):
    half_len = int(np.ceil(n_std * alpha_seconds * fs))
    n = np.arange(-half_len, half_len + 1)
    tt = n / fs
    return np.exp(-(tt**2) / (2 * alpha_seconds**2))

# ============================
# STAGE 2B
# ============================
print("\n" + "="*60)
print("STAGE 2B: Transform Representation Concentration")
print("="*60 + "\n")

print("Gaussian pulse:")
print("-" * 60)

# Gaussian STFT at 3 different window widths
for alpha in alphas:
    win = gaussian_window(alpha, fs, n_std=4)
    f, t_seg, Zxx = signal.stft(f1, fs=fs, window=win, nperseg=len(win),
                                 noverlap=int(0.75*len(win)), boundary=None, padded=False)
    P = np.abs(Zxx)**2
    t_real = t_pulse[0] + t_seg
    sigma_t, sigma_omega, U = compute_2d_tf_concentration(P, t_real, f)
    print(f"  STFT (α={alpha:.2f}s): σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")

# Gaussian CWT at 3 different scale bands
print("\n  CWT at different scale bands:")
for config in cwt_scale_configs:
    center_freq = config["center_freq"]
    bandwidth = config["bandwidth"]
    
    # Narrow frequency band around center
    f_low = center_freq * (1 - bandwidth)
    f_high = center_freq * (1 + bandwidth)
    
    freqs_band = np.geomspace(f_low, f_high, 15)
    scales_band = pywt.frequency2scale(wavelet, freqs_band / fs)
    
    coefs, freqs_actual = pywt.cwt(f1, scales_band, wavelet, sampling_period=dt, method="fft")
    P_cwt = np.abs(coefs)**2
    
    sigma_t, sigma_omega, U = compute_2d_tf_concentration(P_cwt, t_pulse, freqs_actual)
    print(f"  CWT ({config['name']}, f≈{center_freq}Hz ±{int(bandwidth*100)}%, 15 scales):")
    print(f"      σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")

print("\n" + "-"*60)
print("Chirp:")
print("-" * 60)

# Chirp STFT
for alpha in alphas:
    win = gaussian_window(alpha, fs, n_std=4)
    f, t_seg, Zxx = signal.stft(f2, fs=fs, window=win, nperseg=len(win),
                                 noverlap=int(0.75*len(win)), boundary=None, padded=False)
    P = np.abs(Zxx)**2
    t_real = t_chirp[0] + t_seg
    sigma_t, sigma_omega, U = compute_2d_tf_concentration(P, t_real, f)
    print(f"  STFT (α={alpha:.2f}s): σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")

# Chirp CWT 
print("\n  CWT at different scale bands:")
for config in cwt_scale_configs:
    center_freq = config["center_freq"]
    bandwidth = config["bandwidth"]
    
    f_low = center_freq * (1 - bandwidth)
    f_high = center_freq * (1 + bandwidth)
    
    freqs_band = np.geomspace(f_low, f_high, 15)
    scales_band = pywt.frequency2scale(wavelet, freqs_band / fs)
    
    coefs, freqs_actual = pywt.cwt(f2, scales_band, wavelet, sampling_period=dt, method="fft")
    P_cwt = np.abs(coefs)**2
    
    sigma_t, sigma_omega, U = compute_2d_tf_concentration(P_cwt, t_chirp, freqs_actual)
    print(f"  CWT ({config['name']}, f≈{center_freq}Hz ±{int(bandwidth*100)}%, 15 scales):")
    print(f"      σ_t={sigma_t:.4f}s, σ_ω={sigma_omega:.4f}rad/s, U={U:.4f}")