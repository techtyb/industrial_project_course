import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
from scipy.optimize import curve_fit

# Gaussian function
def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

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


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(smoothed_data)
ax1.plot(peaks, smoothed_data[peaks], "rx")
ax1.axhline(y=height_threshold, color='y', linewidth=2)

ax2.plot(smoothed_data, label='Smoothed Data')

for peak in peaks:
    try:
        window = 500
        x_data = np.arange(peak-window, peak+window)
        y_data = smoothed_data[x_data]

        params, _ = curve_fit(gaussian, x_data, y_data, p0=[smoothed_data[peak], peak, window/2])
        fit = gaussian(x_data, *params)
        ax2.plot(x_data, fit, "r-", label=f'Gaussian Fit around {peak}')

    except (RuntimeError, ValueError, IndexError):
        continue

ax1.set_ylim(0, 2800)
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")

ax2.set_ylim(0, 2800)
ax2.set_xlabel("Time")
ax2.set_ylabel("Gaussian Peaks")



plt.show()
