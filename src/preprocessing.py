import numpy as np
import re
import os
import antropy as ant
import matplotlib.pyplot as plt
import antropy as ant
from scipy.signal import welch, hilbert
from scipy.stats import skew, kurtosis


def parse_summary_file(summary_path, subject):
    """
    Parses a summary text file to extract seizure time intervals for a specific subject.
    Returns a dictionary mapping filenames to lists of (start, end) tuples of a seizures.
    Example output:
    {'chb24_01.edf': [(480, 505), (2451, 2476)],'chb24_02.edf': [], 'chb24_03.edf': [(231, 260), (2883, 2908)], ...}
    """
    seizure_info = {}
    # Open the file and read its content
    with open(summary_path, "r") as f:
        content = f.read()

    # Split the content into blocks, each starting with "File Name: "
    file_blocks = content.split("File Name: ")
    for block in file_blocks:
        # Skip the block if it doesn't belong to the target subject (e.g., 'chb01')
        if subject not in block:
            continue

        # Extract the filename from the first line of the block
        filename = block.split("\n")[0].strip()

        # Use regex to find the total number of seizures reported in this file
        num_seizures = int(
            re.search(r"Number of Seizures in File: (\d+)", block).group(1)
        )

        intervals = []
        # If there are seizures, extract their start and end times
        if num_seizures > 0:
            # Seizure        -> literal string
            # (?:\s+\d+)?    -> optional non-capturing group for space and digits (e.g., " 1")
            # \s+Start Time: -> literal string with space
            # \s*(\d+)       -> captured digits (the actual timestamp)
            starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", block)
            ends = re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", block)

            for s, e in zip(starts, ends):
                intervals.append((int(s), int(e)))

        seizure_info[filename] = intervals

    return seizure_info




def _safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def _sample_entropy_safe(x):
    try:
        return ant.sample_entropy(x)
    except:
        return 0.0


def _perm_entropy_safe(x):
    try:
        return ant.perm_entropy(x, normalize=True)
    except:
        return 0.0


def _spectral_entropy_safe(psd_row):
    psd_norm = psd_row / (np.sum(psd_row) + 1e-8)
    psd_norm = psd_norm + 1e-12  # żeby uniknąć log(0)
    return -np.sum(psd_norm * np.log(psd_norm))


def extract_node_features(window, fs):
    """
    window: (n_channels, n_samples)
    return: (n_channels, n_features)
    """

    n_channels, n_samples = window.shape

    # =========================
    # 1. Amplituda / energia
    # =========================
    stds = np.std(window, axis=1)
    rmss = np.sqrt(np.mean(window**2, axis=1))

    d1 = np.diff(window, axis=1)
    line_length = np.sum(np.abs(d1), axis=1)

    # =========================
    # 2. Hjorth
    # =========================
    d2 = np.diff(d1, axis=1)

    m0 = np.var(window, axis=1)
    m2 = np.var(d1, axis=1)
    m4 = np.var(d2, axis=1)

    mobility = np.sqrt(_safe_div(m2, m0))
    complexity = np.sqrt(_safe_div(m4, m2)) / (mobility + 1e-8)

    # =========================
    # 3. Statystyki rozkładu
    # =========================
    skews = skew(window, axis=1)
    kurts = kurtosis(window, axis=1)
    ptp = np.ptp(window, axis=1)

    # =========================
    # 4. Zero crossings
    # =========================
    zero_crossings = np.sum(np.diff(np.sign(window), axis=1) != 0, axis=1)

    # =========================
    # 5. Higuchi FD
    # =========================
    hfds = np.array([ant.higuchi_fd(ch) for ch in window])

    # =========================
    # 6. Teager Energy
    # =========================
    teo = window[:, 1:-1]**2 - window[:, :-2] * window[:, 2:]
    teo_mean = np.mean(teo, axis=1)

    # =========================
    # 7. PSD (Welch)
    # =========================
    freqs, psd = welch(window, fs, nperseg=fs, axis=1)

    total_power = np.trapz(psd, freqs, axis=1)

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        # 'gamma': (30, 45)
    }

    band_powers = []
    for low, high in bands.values():
        idx = (freqs >= low) & (freqs <= high)
        power = np.trapz(psd[:, idx], freqs[idx], axis=1)
        band_powers.append(power)

    band_powers = np.array(band_powers)  # (5, n_channels)

    # =========================
    # 8. Relative band powers
    # =========================
    rel_band_powers = _safe_div(band_powers, total_power)

    # rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma = rel_band_powers
    rel_delta, rel_theta, rel_alpha, rel_beta = rel_band_powers

    # =========================
    # 9. Ratio features
    # =========================
    theta_alpha_ratio = _safe_div(rel_theta, rel_alpha)
    delta_beta_ratio = _safe_div(rel_delta, rel_beta)

    # =========================
    # 10. Entropie
    # =========================
    # sampen = np.array([_sample_entropy_safe(ch) for ch in window])
    perm_entropy = np.array([_perm_entropy_safe(ch) for ch in window])

    spectral_entropy = np.array([
        _spectral_entropy_safe(psd[i]) for i in range(n_channels)
    ])

    # =========================
    # FINAL STACK
    # =========================
    features = np.column_stack([
        # amplituda
        stds, rmss, line_length,

        # Hjorth
        mobility, complexity,

        # statystyka
        skews, kurts, ptp,

        # dynamika
        zero_crossings,

        # nonlinear
        hfds, teo_mean,

        # freq (relative)
        # rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma,
        rel_delta, rel_theta, rel_alpha, rel_beta,

        # ratios
        theta_alpha_ratio, delta_beta_ratio,

        # entropy
        # sampen, perm_entropy, spectral_entropy
        perm_entropy, spectral_entropy
    ])

    return np.nan_to_num(features)

def compute_plv_new(window):
    analytic_signal = hilbert(window, axis=1)
    phase_vectors = analytic_signal / np.abs(analytic_signal)
    
    n_samples = window.shape[1]
    plv_matrix = np.abs(np.dot(phase_vectors, phase_vectors.conj().T)) / n_samples
    
    # Progowanie i usuwanie przekątnej
    np.fill_diagonal(plv_matrix, 0)
    return plv_matrix

def compute_plv(window, threshold="average"):
    """
    Zoptymalizowane PLV (Phase Locking Value) bez pętli.
    """
    # Transformata Hilberta (zoptymalizowana w scipy dla potęg 2)
    analytic_signal = hilbert(window, axis=1)
    # Wyciągamy jednostkowe wektory fazy: e^(i * faza)
    phase_vectors = analytic_signal / np.abs(analytic_signal)
    
    # Średnia różnica faz: 1/S * |sum(exp(i(phi_1 - phi_2)))|
    # To jest równoważne iloczynowi macierzowemu:
    n_samples = window.shape[1]
    plv_matrix = np.abs(np.dot(phase_vectors, phase_vectors.conj().T)) / n_samples
    
    # Progowanie i usuwanie przekątnej
    np.fill_diagonal(plv_matrix, 0)
    if threshold == "average":
        thresh_value = np.mean(plv_matrix)
    elif isinstance(threshold, (int, float)):
        thresh_value = threshold
    else:
        raise ValueError("Nieznany próg dla PLV.")

    plv_matrix[plv_matrix < thresh_value] = 0
    
    return plv_matrix

def compute_corr(window):
    # 1. Oblicz macierz korelacji
    corr = np.corrcoef(window)
    
    # 2. Obsłuż błędy numeryczne (stały sygnał = NaN)
    # Zastępujemy NaN zerami, bo brak zmienności to brak korelacji
    corr = np.nan_to_num(corr)
    
    # 3. Wartość bezwzględna (kluczowa dla montażu bipolarnego)
    corr_abs = np.abs(corr)
    
    # 4. Obsługa przekątnej
    np.fill_diagonal(corr_abs, 0)
    
    # Opcjonalnie: proste progowanie (thresholding)
    # corr_abs[corr_abs < 0.1] = 0 
    
    return corr_abs

def compute_coherence(window):
    """
    window: [N_channels, T]
    """
    fft = np.fft.rfft(window, axis=1)
    
    Sxy = fft[:, None, :] * np.conj(fft[None, :, :])
    Sxx = np.abs(fft[:, None, :])**2
    Syy = np.abs(fft[None, :, :])**2

    coh = np.abs(Sxy)**2 / (Sxx * Syy + 1e-8)

    # średnia po częstotliwościach
    coh = coh.mean(axis=2)

    np.fill_diagonal(coh, 0)
    return coh

def compute_plv_pli(window):
    """
    window: [N_channels, T]
    returns:
        plv: [N, N]
        pli: [N, N]
    """
    analytic_signal = hilbert(window, axis=1)
    
    # jednostkowe wektory fazy (bardziej stabilne niż angle)
    phase_vectors = analytic_signal / np.abs(analytic_signal)

    n_samples = window.shape[1]

    # --- PLV ---
    plv = np.abs(np.dot(phase_vectors, phase_vectors.conj().T)) / n_samples

    # --- PLI ---
    # imag part = sin(delta_phase)
    imag_part = np.imag(
        phase_vectors[:, None, :] * np.conj(phase_vectors[None, :, :])
    )

    pli = np.abs(np.mean(np.sign(imag_part), axis=2))

    # --- cleanup ---
    np.fill_diagonal(plv, 0)
    np.fill_diagonal(pli, 0)

    return plv, pli
