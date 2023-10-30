# -*- coding: utf-8 -*-
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, detrend
from matplotlib.patches import ConnectionPatch

# Moving average function
def moving_average(data, window_size=100):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

def process_peak(x):
    print(x)
    # Values in peak_on_seconds are not particularly accurate.
    # Find the closest indices.
    

tracemalloc.start()

benchmark_signal_file_path = ("Benchmark signal.csv")

the_connection_patches_are_definitely_not_an_eyesore = True
i_want_to_see_the_peak_markers = True

print("\nMemory usage before initial setup")
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

data_path = "Benchmark signal.csv"
samples_per_second = 50000
seconds = 500
data_raw = pd.read_csv(data_path, nrows=samples_per_second * seconds, usecols=['adc2']).values

# Noise data and thresholds
noise_data_path = "just water.csv"
noise_data_raw = pd.read_csv(noise_data_path, usecols=['adc2']).values

# Detrend the noise data
# detrended_noise_data = noise_data_raw

mean_noise, std_noise = noise_data_raw.mean(), noise_data_raw.std()
height_threshold = mean_noise + 5 * std_noise

# detrend actual data
# corrected_data = data_raw
smoothed_data = moving_average(data_raw)

# Calculating the distance value for the peak dection by calculating the average distance between
# each peak in our signal
peaks_without_distance, _ = find_peaks(smoothed_data, height=height_threshold)

distances = np.diff(peaks_without_distance)

# Calculate the average distance
average_distance = np.mean(distances)


peaks, _ = find_peaks(smoothed_data, height=height_threshold, distance=average_distance)
peak_on_seconds = [value/samples_per_second for index, value in enumerate(peaks)]

# fmt: on

# Find indices for small zoom windows
# Signal 1: Tissue
# Signal 2: Noise
# Signal 3: Water
subsignal_index_1 = np.logical_and(66 <= benchmark_time, benchmark_time <= 68)
subsignal_index_2 = np.logical_and(300 <= benchmark_time, benchmark_time <= 302)
subsignal_index_3 = np.logical_and(416 <= benchmark_time, benchmark_time <= 418)

# Extract the signals into their own variables here.
# This mostly makes the code clearer (hopefully)
subtime_1 = benchmark_time[subsignal_index_1]
subtime_2 = benchmark_time[subsignal_index_2]
subtime_3 = benchmark_time[subsignal_index_3]
subsignal_1 = benchmark_signal[subsignal_index_1]
subsignal_2 = benchmark_signal[subsignal_index_2]
subsignal_3 = benchmark_signal[subsignal_index_3]

# Use same y-axis limits for both signal 2 and 3
# Why are the value being divided as integers and multiplied with 10?
# To floor the values to closest ten: the min-max limits for 365 would be 360-370.
subsignal_min = np.min([subsignal_2, subsignal_3]) // 10 * 10
subsignal_max = np.max([subsignal_2, subsignal_3]) // 10 * 10 + 10

# Plot a mosaic. This setting plots one large plot at the top of the figure
# and three smaller plots on the bottom of the figure.
fig, axd = plt.subplot_mosaic(
    [
        ["main", "main", "main"],
        ["lower left", "lower center", "lower right"],
    ],
    figsize=(16, 10),
    layout="constrained",
)

# Plot the benchmark signal on the large plot on top
axd["main"].plot(benchmark_time, benchmark_signal)
axd["main"].set_xlabel("Time [s]")
axd["main"].set_ylabel("Signal [arb]")
axd["main"].set_title("Benchmark signal")
axd["main"].set_xlim(benchmark_time[0], benchmark_time[-1])

# Mark zoom window areas as colored spans in the benchmark signal plot
axd["main"].axvspan(subtime_1[0], subtime_1[-1], 0, 4095, facecolor="g", alpha=0.5)
axd["main"].axvspan(subtime_2[0], subtime_2[-1], 0, 4095, facecolor="r", alpha=0.5)
axd["main"].axvspan(subtime_3[0], subtime_3[-1], 0, 4095, facecolor="b", alpha=0.5)

# Mark noise, tissue and water sections as colored bands:
# red: noise
# green: tissue
# blue: water
axd["main"].axvspan(0, 17, 0, 4095, facecolor="r", alpha=0.1)
axd["main"].axvspan(17, 160, 0, 4095, facecolor="g", alpha=0.1)
axd["main"].axvspan(160, 380, 0, 4095, facecolor="r", alpha=0.1)
axd["main"].axvspan(380, 500, 0, 4095, facecolor="b", alpha=0.1)

# Plot signal 1 in a green background. In these plots, the background
# comes from a semitransparent span.
axd["lower left"].axvspan(
    subtime_1[0], subtime_1[-1], 0, 4095, facecolor="g", alpha=0.25
)
axd["lower left"].plot(subtime_1, subsignal_1)
axd["lower left"].set_xlabel("Time [s]")
axd["lower left"].set_title("Tissue peaks")
axd["lower left"].set_xlim(subtime_1[0], subtime_1[-1])

# Plot signal 2 in a red background
axd["lower center"].plot(subtime_2, subsignal_2)
axd["lower center"].set_xlabel("Time [s]")
axd["lower center"].set_title("Noise peaks")
axd["lower center"].set_xlim(subtime_2[0], subtime_2[-1])
axd["lower center"].set_ylim(subsignal_min, subsignal_max)
axd["lower center"].axvspan(
    subtime_2[0], subtime_2[-1], 0, 4095, facecolor="r", alpha=0.25
)

# Plot signal 3 with a blue background
axd["lower right"].plot(subtime_3, subsignal_3)
axd["lower right"].set_xlabel("Time [s]")
axd["lower right"].set_title("Water peak")
axd["lower right"].set_xlim(subtime_3[0], subtime_3[-1])
axd["lower right"].set_ylim(subsignal_min, subsignal_max)
axd["lower right"].axvspan(
    subtime_3[0], subtime_3[-1], 0, 4095, facecolor="b", alpha=0.25
)


# Inter-plot annotation arrows. These connect the zoom window marks in the
# main plot to the lower plots.
if the_connection_patches_are_definitely_not_an_eyesore:
    con_1 = ConnectionPatch(
        xyA=(subtime_1[0] + (subtime_1[-1] - subtime_1[0]) / 2, np.min(subsignal_1)),
        coordsA=axd["main"].transData,
        xyB=(subtime_1[0] + (subtime_1[-1] - subtime_1[0]) / 2, np.max(subsignal_1)),
        coordsB=axd["lower left"].transData,
        arrowstyle="<->",
    )
    con_2 = ConnectionPatch(
        xyA=(subtime_2[0] + (subtime_2[-1] - subtime_2[0]) / 2, np.min(subsignal_2)),
        coordsA=axd["main"].transData,
        xyB=(subtime_2[0] + (subtime_2[-1] - subtime_2[0]) / 2, np.max(subsignal_2)),
        coordsB=axd["lower center"].transData,
        arrowstyle="<->",
    )
    con_3 = ConnectionPatch(
        xyA=(subtime_3[0] + (subtime_3[-1] - subtime_3[0]) / 2, np.min(subsignal_3)),
        coordsA=axd["main"].transData,
        xyB=(subtime_3[0] + (subtime_3[-1] - subtime_3[0]) / 2, np.max(subsignal_3)),
        coordsB=axd["lower right"].transData,
        arrowstyle="<->",
    )
    fig.add_artist(con_1)
    fig.add_artist(con_2)
    fig.add_artist(con_3)

print(len(peak_on_seconds))
# Annotate tissue peaks in the main plot and in the lower left plot
if i_want_to_see_the_peak_markers:
    for x in peak_on_seconds:
        equal_or_greater = x - 1 / 50000 <= benchmark_time
        equal_or_less = benchmark_time <= x + 1 / 50000
        idx = np.logical_and(equal_or_greater, equal_or_less)
        idx = np.flatnonzero(idx)[0]  # Pick one

        # Get suitable positions for the elements
        x_arrow_head = benchmark_time[idx]
        y_arrow_head = benchmark_signal[idx - 10000 : idx + 10000].max()
        x_text = x_arrow_head
        y_text = y_arrow_head + 100

        # Annotate!
        axd["main"].annotate(
            "",
            xy=(x_arrow_head, y_arrow_head),
            xytext=(x_text, y_text),
            arrowprops=dict(arrowstyle="->"),
        )

        # Should we annotate in lower left plot?
        if subtime_1[0] <= x_arrow_head <= subtime_1[-1]:
            # Again, find suitable position:
            sub_geq = x - 1 / 50000 <= subtime_1
            sub_leq = subtime_1 <= x + 1 / 50000
            sub_idx = np.logical_and(sub_geq, sub_leq)
            sub_idx = np.flatnonzero(sub_idx)[0]

            sub_x_arrow = subtime_1[sub_idx]
            sub_y_arrow = subsignal_1[sub_idx - 1000 : sub_idx + 1000].max()
            sub_x_text = sub_x_arrow
            sub_y_text = sub_y_arrow + 25

            axd["lower left"].annotate(
                "",
                xy=(sub_x_arrow, sub_y_arrow),
                xytext=(sub_x_text, sub_y_text),
                arrowprops=dict(arrowstyle="->"),
            )
            
print("\nMemory usage after initial setup")
trace_first_size, trace_first_peak = tracemalloc.get_traced_memory()
print(f"Traced size: {trace_first_size / 1024 / 1024:.2f} MB")
print(f"Traced peak: {trace_first_peak / 1024 / 1024:.2f} MB")
total_memory = sum(x.size for x in tracemalloc.take_snapshot().statistics("lineno"))
print(f"Total size: {total_memory / 1024 / 1024:.2f} MB")
tracemalloc.reset_peak()

# %% Implement your algorithm and plotting here
# Feel free to use more blocks to divide your code into logical sections.
#

# Let's discard some of the temporary variables
del subsignal_index_1, subsignal_index_2, subsignal_index_3
del subtime_1, subtime_2, subtime_3
del subsignal_1, subsignal_2, subsignal_3
del subsignal_min, subsignal_max
del con_1, con_2, con_3
# del equal_or_greater, equal_or_less, idx
# del sub_leq, sub_idx, sub_x_arrow, sub_y_arrow, sub_x_text, sub_y_text
del i_want_to_see_the_peak_markers
del the_connection_patches_are_definitely_not_an_eyesore
# del x
# del x_arrow_head, y_arrow_head, x_text, y_text

print("\nMemory usage after initial setup")
trace_first_size, trace_first_peak = tracemalloc.get_traced_memory()
print(f"Traced size: {trace_first_size / 1024 / 1024:.2f} MB")
print(f"Traced peak: {trace_first_peak / 1024 / 1024:.2f} MB")
total_memory = sum(x.size for x in tracemalloc.take_snapshot().statistics("lineno"))
print(f"Total size: {total_memory / 1024 / 1024:.2f} MB")
tracemalloc.reset_peak()
plt.show()

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
