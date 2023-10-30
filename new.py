# -*- coding: utf-8 -*-
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, detrend
from matplotlib.patches import ConnectionPatch

tracemalloc.start()

benchmark_signal_file_path = ("Benchmark signal.csv")

the_connection_patches_are_definitely_not_an_eyesore = True
i_want_to_see_the_peak_markers = True

print("\nMemory usage after initial setup")
trace_first_size, trace_first_peak = tracemalloc.get_traced_memory()
print(f"Traced size: {trace_first_size / 1024 / 1024:.2f} MB")
print(f"Traced peak: {trace_first_peak / 1024 / 1024:.2f} MB")
total_memory = sum(x.size for x in tracemalloc.take_snapshot().statistics("lineno"))
print(f"Total size: {total_memory / 1024 / 1024:.2f} MB")
tracemalloc.reset_peak()

# %% Read the benchmark data
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!                                                         !!!
# !!! The groups are not allowed to change code in this block !!!
# !!!                                                         !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

benchmark_signal = pd.read_csv(
    benchmark_signal_file_path, usecols=[1], dtype=np.uint16
).to_numpy()[:, 0]

benchmark_time = np.linspace(
    0, benchmark_signal.shape[0] / 50000, benchmark_signal.shape[0]
)

print("\nMemory usage after initial setup")
trace_first_size, trace_first_peak = tracemalloc.get_traced_memory()
print(f"Traced size: {trace_first_size / 1024 / 1024:.2f} MB")
print(f"Traced peak: {trace_first_peak / 1024 / 1024:.2f} MB")
total_memory = sum(x.size for x in tracemalloc.take_snapshot().statistics("lineno"))
print(f"Total size: {total_memory / 1024 / 1024:.2f} MB")
tracemalloc.reset_peak()

# %% Plot the benchmark signal
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!                                                         !!!
# !!! The groups are not allowed to change code in this block !!!
# !!!                                                         !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# fmt: off

# Some peaks that should be detected in the tissue suction section of the
# signal. This includes visually trivial peaks and some more challenging
# likely merged peaks.
#
# Note that no peaks should be detected in the noise area. Peaks in the water
# area not considered interesting and are not included in this code template,
# but you may annotate those also with a different color.

# Moving average function
def moving_average(data, window_size=100):
    return data.rolling(window=window_size).mean()

data_path = "Benchmark signal.csv"
samples_per_second = 50000
seconds = 50
data_df = pd.read_csv(data_path, nrows=samples_per_second * seconds)
data_raw = data_df['adc2'].values

# Noise data and thresholds
noise_data_path = "just water.csv"
noise_data_df = pd.read_csv(noise_data_path)
noise_data_raw = noise_data_df['adc2']

# Detrend the noise data
detrended_noise_data = noise_data_raw

mean_noise, std_noise = detrended_noise_data.mean(), detrended_noise_data.std()
height_threshold = mean_noise + 5 * std_noise

# detrend actual data
corrected_data = data_raw
smoothed_data = moving_average(pd.Series(corrected_data)).values
peaks, _ = find_peaks(smoothed_data, height=height_threshold)
peak_on_seconds = [value/50000 for index, value in enumerate(data)]



# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# ax1.plot(smoothed_data)
# ax1.plot(peaks, smoothed_data[peaks], "rx")
# ax1.axhline(y=height_threshold, color='y', linewidth=2)

# ax2.plot(smoothed_data, label='Smoothed Data')

# for peak in peaks:
#     try:
#         window = 500
#         x_data = np.arange(peak-window, peak+window)
#         y_data = smoothed_data[x_data]

#         params, _ = curve_fit(gaussian, x_data, y_data, p0=[smoothed_data[peak], peak, window/2])
#         fit = gaussian(x_data, *params)
#         ax2.plot(x_data, fit, "r-", label=f'Gaussian Fit around {peak}')

#     except (RuntimeError, ValueError, IndexError):
#         continue

# ax1.set_ylim(0, 2800)
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Amplitude")

# ax2.set_ylim(0, 2800)
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Gaussian Peaks")

# plt.show()
