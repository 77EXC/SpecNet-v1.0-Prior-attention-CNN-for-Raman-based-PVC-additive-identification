import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- 1. Load the processed data ---
JSON_PATH = "raman_dataset_processed_interpolated.json"

print(f"Loading data from '{JSON_PATH}'...")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract spectra, wavenumbers, and sample IDs
all_spectra = np.array([sample['raman_spectrum']['intensity_processed'] for sample in data])
wavenumbers = np.array(data[0]['raman_spectrum']['wavenumber_processed'])
sample_ids = [sample['sample_id'] for sample in data] # Load sample IDs
print(f"Data loaded successfully. Shape: {all_spectra.shape}")

# --- 2. Calculate point-wise variance ---
# np.var(axis=0) calculates the variance for each column (each wavenumber point)
variance_spectrum = np.var(all_spectra, axis=0)

# --- 3. Find peaks in the variance spectrum to identify important regions ---
# 尽管我们不绘制这些峰和波段，但保留这部分逻辑以便您可以检查它们的值
# find_peaks parameters can be tuned to filter the peaks
mean_variance = np.mean(variance_spectrum)
peaks, properties = find_peaks(
    variance_spectrum, 
    height=mean_variance,  # Only consider peaks higher than the mean variance
    distance=20,           # Peaks must be at least 20 data points apart
    width=10,              # Peaks must be at least 10 data points wide
    rel_height=0.5         # Width is measured at 50% of the peak height
)

# --- 4. Extract band boundaries from the found peaks ---
band_boundaries_indices = []
recommended_bands = {}
for i, (left_idx, right_idx) in enumerate(zip(properties['left_ips'], properties['right_ips'])):
    start_idx = int(np.floor(left_idx))
    end_idx = int(np.ceil(right_idx))
    start_wn = wavenumbers[start_idx]
    end_wn = wavenumbers[end_idx]
    
    band_boundaries_indices.append((start_idx, end_idx))
    recommended_bands[f'band_{i+1}'] = (round(start_wn), round(end_wn))

print(f"\nAutomatically detected {len(peaks)} important regions from the variance spectrum.")
print("\n--- Recommended Wavenumber Bands (can be copied to the training script) ---")
print("WAVE_BANDS = {")
for name, (start, end) in recommended_bands.items():
    print(f"    '{name}': ({start}, {end}),")
print("}")


# --- 5. Visualize the analysis results ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot 1: Plot a few random spectra as a reference
axs[0].set_title("Randomly Sampled Spectra", fontsize=16)
num_samples_to_plot = 3
random_indices = np.random.choice(len(all_spectra), num_samples_to_plot, replace=False)

for i in random_indices:
    axs[0].plot(wavenumbers, all_spectra[i], alpha=0.8, label=sample_ids[i])

axs[0].set_ylabel("Normalized Intensity", fontsize=12)
axs[0].legend(loc='upper right')

# Plot 2: 只绘制方差谱线，不显示峰值标记和波段高亮
axs[1].set_title("Spectral Variance", fontsize=16) # 更改标题
axs[1].plot(wavenumbers, variance_spectrum, color='black', label='Point-wise Variance')

axs[1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
axs[1].set_ylabel("Variance", fontsize=12)
axs[1].legend(loc='upper right') # 此时图例只会显示 'Point-wise Variance'
axs[1].set_yscale('log') # 使用对数刻度以更好地显示微小方差变化

plt.tight_layout()
plt.savefig("wavenumber_bands_analysis.png", dpi=150)
plt.show()

print("\nAnalysis plot saved as 'wavenumber_bands_analysis.png'")