import os
import numpy as np
import mne
import torch
import gc
import pandas as pd 
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path
from typing import Union
from collections import Counter
from scipy.signal import savgol_filter

from src.config import PROCESSED_DIR, RAW_DIR, COMMON_CHANNELS
from src.preprocessing import parse_summary_file, extract_node_features, compute_plv

import random

def set_seed(seed: int = 42):
    # Podstawowy Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # dla rozwiązań wieloprocesorowych (multi-GPU)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seed set to: {seed}")

set_seed(42)


WINDOW_SIZE_SEC = 6
SAMPLING_RATE = 256 # Hz
SAMPLES_PER_WINDOW = WINDOW_SIZE_SEC * SAMPLING_RATE
OVERLAP_STEP_SEC = 0.5 # WINDOW_SIZE_SEC - OVERLAP_STEP_SEC = overlap kolejnych okien w trakcie napadu
EDGE_THRESHOLD = "average"  # "average" or float value - Dla PLV przy tworzeniu krawędzi grafu (na podstawie artykułu: https://ieeexplore.ieee.org/document/10635821)
LINE_FREQ = 60  # or 50 Hz depending on the region, dane pochodzą z USA, więc 60 Hz
PREICTAL_DURATION_SEC = 10 * 60  # Do nastepnych eksperymentów zmniejszamy do 5 minut (lub ewentualnie 10)
BUFFER_SEC = 15                 # Bufor przed i po napadzie do usunięcia (na podstawie artykułu: https://ieeexplore.ieee.org/document/10635821)
LABEL_INTERICTAL = 0
LABEL_ICTAL = 1
LABEL_PREICTAL = 2
MIN_VAR = 0.1   # mikrowolty^2


import warnings

# Wycisza ostrzeżenia o duplikujących się nazwach kanałów - problem zaadresowany w przetwarzaniu
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Channel names are not unique.*")
# Wycisza ostrzeżenie o brakującym skalowaniu - pojawia się przy zbędnych kanałach, które są usuwane w trakcie przetwarzania
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Scaling factor is not defined.*")


def process_data(subject: str, data_folder: Union[str, Path], summary_file: Union[str, Path]):
    seizure_map = parse_summary_file(summary_file, subject)
    overlap_samples = int(OVERLAP_STEP_SEC * SAMPLING_RATE)
    data_list = []

    # --- DIAGNOSTYKA: Liczniki ---
    stats = {
        'total_windows': 0,
        'skipped_artifact': 0,
        'skipped_buffer': 0,
        'labels': Counter()
    }


    for filename in tqdm(seizure_map.keys(), desc=f"EDF files of {subject}"):
        path = os.path.join(data_folder, filename)
        if not os.path.exists(path):
            continue

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        # NOTE Diagnostyka - Wypisanie listy nazw kanałów oznaczonych jako złe
        bads = raw.info['bads']
        if len(bads) > 0:
            print(f"Złe kanały: {bads}")

        sfreq = raw.info['sfreq']
        if sfreq != SAMPLING_RATE:
            print(f"Plik {filename} ma niestandardową częstotliwość próbkowania: {sfreq} Hz. Oczekiwano: {SAMPLING_RATE} Hz.")



        # 1. STANDARYZACJA I UNIFIKACJA NAZW KANAŁÓW
        raw.rename_channels(lambda x: x.strip().upper())

        # Tworzymy mapowanie dla wariantów T8-P8 - powtarzający się kanał
        # Jeśli w pliku automatycznie pojawi się T8-P8-0 lub T8-P8-1, zmieniamy na T8-P8

        mapping = {'T8-P8-0': 'T8-P8'}
        if 'T8-P8-0' in raw.ch_names:
            raw.rename_channels(mapping)

        try:
            # pick_channels wymusza konkretną kolejność i zestaw elektrod
            raw.pick(COMMON_CHANNELS)
        except ValueError as e:
            print(f"Pomijam plik {filename}: brak wymaganych kanałów. Błąd: {e}")
            raw.close(); del raw; gc.collect()
            continue
        
        # notch filter -> bandpass -> average reference
        raw.notch_filter(freqs=LINE_FREQ, n_jobs=-1, verbose=False)

        raw.filter(1.0, 30, method='iir', n_jobs=-1, verbose=False)

        raw.set_eeg_reference("average")


        data = raw.get_data() * 1e6 # konwersja do mikrovoltów - dla stabilności numerycznej

        # Parametry Savgol: 
        window_len = 11 
        poly_order = 3

        # smoothing filter
        data = savgol_filter(data, window_length=window_len, polyorder=poly_order, axis=-1)


        n_samples = data.shape[1]
        intervals = seizure_map[filename]

        idx = 0

        # Logika wycinania okien o długości WINDOW_SIZE_SEC z odpowiednimi etykietami
        while idx + SAMPLES_PER_WINDOW <= n_samples:
            start_sec = idx / SAMPLING_RATE
            mid_sec = start_sec + WINDOW_SIZE_SEC / 2
            label = None

            is_seizure = any(s <= mid_sec <= e for s, e in intervals) # conajmniej 50 % okna w napadzie

            # Nadawanie etykiet
            if is_seizure:  
                label = LABEL_ICTAL
            else:
                # downsampling na poziomie próbek nienapadowych
                a = np.random.rand() 
                if a < 0.70:  # Logujemy tylko 30% takich plików
                    idx += SAMPLES_PER_WINDOW
                    continue

                # 2. Sprawdź czy okno wpada w bufor bezpieczeństwa (np. 15s przed/po napadzie)
                # Usuwamy okna, które dotykają granicy napadu
                is_in_buffer = any((s - BUFFER_SEC <= mid_sec <= s) or (e <= mid_sec <= e + BUFFER_SEC) 
                                   for s, e in intervals)
                
                if is_in_buffer:
                    label = None # Do pominięcia
                    stats['skipped_buffer'] += 1
                    print(f"Pomijam okno w {filename} od {start_sec:.1f}s z powodu buforu bezpieczeństwa.")
                else:
                    # 3. Sprawdź czy to Preictal
                    is_preictal = any(s - PREICTAL_DURATION_SEC <= mid_sec <= s - BUFFER_SEC 
                                      for s, e in intervals)
                    if is_preictal:
                        label = LABEL_PREICTAL
                    else:
                        label = LABEL_INTERICTAL


            # 2. SANITY CHECK: Artefakty (bardzo mała wariancja -> płaski kanał -> prawdopodobny błąd)
            window = data[:, idx:idx + SAMPLES_PER_WINDOW]

            is_flat = np.any(np.var(window, axis=1) < MIN_VAR)

            if label is None or is_flat:
                if label is not None:
                    stats['skipped_artifact'] += 1

                    print(f"Pomijam okno w {filename} od {start_sec:.1f}s z powodu flat-line. Label {label}")
                idx += overlap_samples if is_seizure else SAMPLES_PER_WINDOW
                continue

            # 3. EKSTRAKCJA CECH I TWORZENIE GRAFU
            node_x = extract_node_features(window, SAMPLING_RATE) # tworzenie cech węzłów
            adj = compute_plv(window, EDGE_THRESHOLD) # tworzenie macierzy sąsiedztwa za pomocą PLV - krawędzie grafu

            # Konwersja na rzadki format (Sparse)
            adj_torch = torch.from_numpy(adj).float()
            edge_index = adj_torch.nonzero().t().contiguous()
            edge_attr = adj_torch[edge_index[0], edge_index[1]]

            pyg_graph = Data(
                x=torch.from_numpy(node_x).float(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long)
            )
            data_list.append(pyg_graph)
            stats['labels'][label] += 1

            # logika przesuwania okna - overlap w zależności od etykiety
            if label == LABEL_ICTAL:
                idx += overlap_samples
            elif label == LABEL_PREICTAL:
                idx += SAMPLES_PER_WINDOW
            else:
                idx += SAMPLES_PER_WINDOW

        raw.close(); del raw; del data; gc.collect()

    # # --- 4. Sanity check ---
    raw_features = torch.cat([g.x for g in data_list], dim=0)
    print(f"\n--- SANITY CHECK: {subject} ---")
    print(f"Rozkład etykiet: {dict(stats['labels'])}")
    print(f"Pominięte okna (bufor): {stats['skipped_buffer']}")
    print(f"Pominięte okna (artefakty): {stats['skipped_artifact']}")
    
    df_stats = pd.DataFrame({
        # 'Feature': feat_names,
        'Mean_Post': raw_features.mean(dim=0).numpy(),
        'Std_Post': raw_features.std(dim=0).numpy(),
        'Min_Post': raw_features.min(dim=0).values.numpy(),
        'Max_Post': raw_features.max(dim=0).values.numpy()
    })
    print("\nStatystyki cech przed normalizacją:")
    print(df_stats.to_string(index=False))
    
    torch.save(data_list, PROCESSED_DIR / f"patient_{subject}.pt")
    print(f"Zapisano {len(data_list)} grafów dla pacjenta {subject}.")


if __name__ == "__main__":
    subjects_to_process = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]   

    for SUBJECT in tqdm(subjects_to_process):
        DATA_FOLDER = RAW_DIR / SUBJECT
        SUMMARY_FILE = DATA_FOLDER / f"{SUBJECT}-summary.txt"

        process_data(SUBJECT, DATA_FOLDER, SUMMARY_FILE)
        gc.collect()
