import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def psd2time(freq, psd, t_max, dt, seed=None):
    """
    Genera un time-history da una PSD unilaterale definita su freq.

    Parameters
    ----------
    freq : array_like
        Vettore delle frequenze (Hz), deve cominciare da 0 e arrivare a f_max.
    psd : array_like
        Valori di PSD(f) in (unit^2/Hz) corrispondenti a freq.
    t_max : float
        Durata del segnale in secondi.
    dt : float
        Passo di campionamento in secondi.
    seed : int, optional
        Seed per la generazione di fasi casuali reproducibili.

    Returns
    -------
    t : ndarray
        Vettore tempi da 0 a t_max - dt.
    x : ndarray
        Segnale nel dominio del tempo.
    """
    n_samples = int(np.floor(t_max / dt))
    df = 1.0 / (n_samples * dt)
    f_rfft = np.arange(0, n_samples // 2 + 1) * df

    psd_interp = interp.interp1d(
        freq,
        psd,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    psd_values = psd_interp(f_rfft)

    if seed is not None:
        np.random.seed(seed)

    amp = np.sqrt(2.0 * psd_values * df)
    amp[0] = np.sqrt(psd_values[0] * df)
    if n_samples % 2 == 0:
        amp[-1] = np.sqrt(psd_values[-1] * df)

    phase = np.random.uniform(0.0, 2.0 * np.pi, size=amp.shape)
    phase[0] = 0.0
    if n_samples % 2 == 0:
        phase[-1] = 0.0

    spectrum = amp * np.exp(1j * phase)
    x = np.fft.irfft(spectrum, n=n_samples)

    t = np.arange(n_samples) * dt
    return t, x


# Carica la PSD da file (due colonne: freq, PSD).
data = np.loadtxt("PSD_50.txt", delimiter=None)
freq = data[:, 0]
psd_values = data[:, 1]

t_max = 20.0
f_max = freq.max()
dt = 1.0 / (20 * f_max)

t, x = psd2time(freq,psd_values,t_max,dt,seed=42,)

rms_theoretical = np.sqrt(np.trapz(psd_values, freq))
rms_effective = np.sqrt(np.mean(x**2))
scale_factor = rms_theoretical / rms_effective
x *= scale_factor

# Plot PSD and time history.
plt.figure()
plt.plot(freq, psd_values, label="PSD input")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (N²/Hz)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(t, x, "r")
plt.xlabel("Time (s)")
plt.ylabel("Signal (N)")
plt.grid(True)
plt.show()

# Salvataggio del segnale in file TXT.
data_out = np.column_stack((t, x))
# np.savetxt("signal_python.txt",data_out,delimiter="\t",fmt="%.6f",)