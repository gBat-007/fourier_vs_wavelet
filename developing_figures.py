import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

# Settings
fs = 400.0
dt = 1.0 / fs

# Define signals
t_pulse = np.linspace(-5, 5, 4000)
f1 = np.exp(-(t_pulse**2)/2)

t_chirp = np.arange(0.0, 10.0, dt)
f2 = np.cos(t_chirp**2)

print("Generating figures...")

# STFT of Gaussian pulse
alpha = 0.50
half_len = int(np.ceil(4 * alpha * fs))
n = np.arange(-half_len, half_len + 1)
tt = n / fs
win = np.exp(-(tt**2) / (2 * alpha**2))

f, t, Zxx = signal.stft(f1, fs=fs, window=win, nperseg=len(win),
                         noverlap=int(0.75*len(win)), boundary=None, padded=False)

plt.figure(figsize=(7, 5))
plt.pcolormesh(t_pulse[0] + t, f, np.abs(Zxx)**2, shading='gouraud', cmap='hot')
plt.colorbar(label='Energy Density')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Frequency (Hz)', fontsize=12)
plt.title('STFT: Gaussian Pulse ($\\alpha=0.50$ s, $U=0.831$)', fontsize=12)
plt.ylim([0, 5])
plt.tight_layout()
plt.savefig('stft_gaussian_pulse.png', dpi=300, bbox_inches='tight')
print("✓ Saved: stft_gaussian_pulse.png")
plt.close()

#CWT of Gaussian pulse (2 Hz band)
freqs_band = np.geomspace(1.4, 2.6, 15)
scales_band = pywt.frequency2scale('cmor1.5-1.0', freqs_band / fs)
coefs, freqs_actual = pywt.cwt(f1, scales_band, 'cmor1.5-1.0', sampling_period=dt, method='fft')

plt.figure(figsize=(7, 5))
plt.pcolormesh(t_pulse, freqs_actual, np.abs(coefs)**2, shading='gouraud', cmap='hot')
plt.colorbar(label='Energy Density')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Frequency (Hz)', fontsize=12)
plt.title('CWT: Gaussian Pulse (2 Hz band, $U=1.402$)', fontsize=12)
plt.tight_layout()
plt.savefig('cwt_gaussian_pulse.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cwt_gaussian_pulse.png")
plt.close()

# STFT of chirp
f, t, Zxx = signal.stft(f2, fs=fs, window=win, nperseg=len(win),
                         noverlap=int(0.75*len(win)), boundary=None, padded=False)

plt.figure(figsize=(7, 5))
plt.pcolormesh(t, f, np.abs(Zxx)**2, shading='gouraud', cmap='hot')
plt.colorbar(label='Energy Density')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Frequency (Hz)', fontsize=12)
plt.title('STFT: Chirp ($\\alpha=0.50$ s, $U=6.632$)', fontsize=12)
plt.ylim([0, 10])
plt.tight_layout()
plt.savefig('stft_chirp.png', dpi=300, bbox_inches='tight')
print("✓ Saved: stft_chirp.png")
plt.close()

# CWT of chirp (2 Hz band)
coefs, freqs_actual = pywt.cwt(f2, scales_band, 'cmor1.5-1.0', sampling_period=dt, method='fft')

plt.figure(figsize=(7, 5))
plt.pcolormesh(t_chirp, freqs_actual, np.abs(coefs)**2, shading='gouraud', cmap='hot')
plt.colorbar(label='Energy Density')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Frequency (Hz)', fontsize=12)
plt.title('CWT: Chirp (2 Hz band, $U=3.471$)', fontsize=12)
plt.tight_layout()
plt.savefig('cwt_chirp.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cwt_chirp.png")
plt.close()

# Parameter comparison (3 alpha values for chirp)
fig, axes = plt.subplots(3, 1, figsize=(8, 9))
alphas = [0.10, 0.25, 0.50]
U_values = [22.27, 12.75, 6.632]

for idx, (alpha, U) in enumerate(zip(alphas, U_values)):
    half_len = int(np.ceil(4 * alpha * fs))
    n = np.arange(-half_len, half_len + 1)
    tt = n / fs
    win = np.exp(-(tt**2) / (2 * alpha**2))
    
    f, t, Zxx = signal.stft(f2, fs=fs, window=win, nperseg=len(win),
                             noverlap=int(0.75*len(win)), boundary=None, padded=False)
    
    im = axes[idx].pcolormesh(t, f, np.abs(Zxx)**2, shading='gouraud', cmap='hot')
    axes[idx].set_ylabel('Frequency (Hz)', fontsize=11)
    axes[idx].set_title(f'$\\alpha = {alpha:.2f}$ s, $U = {U:.2f}$', fontsize=12)
    axes[idx].set_ylim([0, 10])
    plt.colorbar(im, ax=axes[idx], label='Energy')

axes[2].set_xlabel('Time (s)', fontsize=11)
plt.tight_layout()
plt.savefig('parameter_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: parameter_comparison.png")
plt.close()