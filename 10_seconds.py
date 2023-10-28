import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend
from scipy.optimize import curve_fit

# Gaussian function
def gaussian(x, amp, mean, std_dev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))


data_path = "Benchmark signal.csv"
samples_per_second = 50000
ten_seconds_data = pd.read_csv(data_path, nrows=samples_per_second*50)['adc2'].values


ten_seconds_data = detrend(ten_seconds_data)


noise_data_path = "just water.csv"
noise_data_df = pd.read_csv(noise_data_path)
noise_data_raw = detrend(noise_data_df['adc2'].values)  
mean_noise, std_noise = noise_data_raw.mean(), noise_data_raw.std()

height_threshold = mean_noise + 3 * std_noise

peaks, _ = find_peaks(ten_seconds_data, height=height_threshold)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(ten_seconds_data, label='Detrended Data')
ax1.plot(peaks, ten_seconds_data[peaks], "rx", label='Detected Peaks')
ax1.axhline(y=height_threshold, color='y', linewidth=2, label='Threshold')
ax1.set_title('Detrended Data with Detected Peaks')
ax1.set_ylabel('Amplitude')
ax1.legend()

ax2.plot(ten_seconds_data, label='Detrended Data')
for peak in peaks:
    try:
        window = 50
        x_data = np.arange(peak-window, peak+window)
        y_data = ten_seconds_data[x_data]
        
        
        amp_init = np.max(y_data)
        mean_init = peak
        std_dev_init = window/3
        
        params, _ = curve_fit(gaussian, x_data, y_data, p0=[amp_init, mean_init, std_dev_init])
        fit = gaussian(x_data, *params)
        ax2.plot(x_data, fit, "r-", label=f'Gaussian Fit around peak {peak}')
        
    except (RuntimeError, ValueError, IndexError):
        continue

ax2.set_title('Gaussian Peaks Fitting')
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude')
ax2.legend()

plt.tight_layout()
plt.show()
