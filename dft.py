import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -----------------------------
# 1. Load data
# -----------------------------
JSON_PATH = "raman_dataset_processed_interpolated.json"
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_spectra = np.array([sample['raman_spectrum']['intensity_processed'] for sample in data])
wavenumbers = np.array(data[0]['raman_spectrum']['wavenumber_processed'])
variance_spectrum = np.var(all_spectra, axis=0)

# -----------------------------
# 2. Define prior regions (your logic)
# -----------------------------
print("Using prior chemical regions for guidance:")
prior_regions = [
    {"name": "fingerprint",   "start_wn": 0,      "end_wn": 1500},
    {"name": "silent",        "start_wn": 1500,   "end_wn": 2700},
    {"name": "ch_stretch",    "start_wn": 2700,   "end_wn": wavenumbers[-1]}
]

# -----------------------------
# 3. Find automatic split points using variance拐点
# -----------------------------
def find拐点_after_last_peak(start_wn, end_wn, threshold=0.4, min_valley=0.05):
    """
    在 [start_wn, end_wn] 区间内：
    1. 找到所有 variance > threshold 的峰；
    2. 取最后一个峰的位置；
    3. 从该峰向右，找第一个 valley（variance < min_valley）；
    4. 再从 valley 向右，找第一个上升拐点（导数由负变正）；
    5. 返回该拐点对应的 wavenumber 和 index。
    """
    start_idx = np.searchsorted(wavenumbers, start_wn, side='left')
    end_idx = np.searchsorted(wavenumbers, end_wn, side='right')
    sub_wn = wavenumbers[start_idx:end_idx]
    sub_var = variance_spectrum[start_idx:end_idx]
    
    # Step 1: Find peaks above threshold
    peak_indices_local, _ = find_peaks(sub_var, height=threshold)
    if len(peak_indices_local) == 0:
        print(f"⚠️ No peaks above {threshold} in [{start_wn}, {end_wn}]")
        return None, None
    
    last_peak_local = peak_indices_local[-1]  # 最后一个峰（在 sub_var 中的索引）
    last_peak_global = start_idx + last_peak_local
    print(f"  Last peak at wavenumber: {wavenumbers[last_peak_global]:.1f} cm⁻¹")

    # Step 2: Look AFTER the last peak
    after_peak_var = sub_var[last_peak_local:]  # shape: (M,)
    after_peak_wn = sub_wn[last_peak_local:]

    # Find first valley (variance < min_valley)
    valley_mask = after_peak_var < min_valley
    if not np.any(valley_mask):
        print(f"  ⚠️ No valley below {min_valley} found after peak.")
        return None, None
    
    valley_offset = np.where(valley_mask)[0][0]  # 相对于 after_peak_var 的偏移
    valley_global_idx = start_idx + last_peak_local + valley_offset
    print(f"  Valley starts at: {wavenumbers[valley_global_idx]:.1f} cm⁻¹")

    # Step 3: From valley onward, find first rising point (diff > 0)
    after_valley_var = after_peak_var[valley_offset:]
    if len(after_valley_var) < 2:
        print("  ⚠️ Not enough points after valley.")
        return None, None

    diff = np.diff(after_valley_var)  # length = len(after_valley_var) - 1
    rising_in_after_valley = np.where(diff > 0)[0]

    if len(rising_in_after_valley) == 0:
        print("  ⚠️ No rising point found after valley.")
        return None, None

    # First rising point is at: valley_offset + rising_in_after_valley[0] + 1?
    # But for simplicity, we take the point where rise STARTS: index = valley_offset + rising_in_after_valley[0]
    rise_offset_in_after_peak = valley_offset + rising_in_after_valley[0]
    rise_global_idx = start_idx + last_peak_local + rise_offset_in_after_peak

    print(f"  First rising point (拐点) at: {wavenumbers[rise_global_idx]:.1f} cm⁻¹")

    return wavenumbers[rise_global_idx], rise_global_idx

# -----------------------------
# 4. Auto-detect split points
# -----------------------------
print("\n🔍 Automatically detecting split points from variance spectrum...")

# Split 1: between fingerprint and silent region (after last peak in 1200-2000)
split1_wn, split1_idx = find拐点_after_last_peak(
    start_wn=1200,
    end_wn=2000,
    threshold=0.04,
    min_valley=0.02
)

if split1_wn is None:
    split1_wn = 1500  # fallback
    split1_idx = np.searchsorted(wavenumbers, split1_wn, side='left')
    print(f"  Using fallback: {split1_wn:.1f} cm⁻¹")

# Split 2: between silent and CH-stretch (before big peak around 2900)
split2_wn, split2_idx = find拐点_after_last_peak(
    start_wn=2500,
    end_wn=2800,
    threshold=0.04,
    min_valley=0.02
)

if split2_wn is None:
    split2_wn = 2700  # fallback
    split2_idx = np.searchsorted(wavenumbers, split2_wn, side='left')
    print(f"  Using fallback: {split2_wn:.1f} cm⁻¹")

print(f"\n✅ Final split points:")
print(f"  Split 1 (fingerprint/silent): {split1_wn:.1f} cm⁻¹")
print(f"  Split 2 (silent/CH-stretch): {split2_wn:.1f} cm⁻¹")

# -----------------------------
# 5. Define segments based on auto-split points
# -----------------------------
segment_indices = [
    (0, split1_idx),             # segment 1: fingerprint
    (split1_idx, split2_idx),    # segment 2: silent
    (split2_idx, len(wavenumbers))  # segment 3: CH-stretch
]

FULL_SEGMENTS = {}
for i, (s, e) in enumerate(segment_indices):
    wn_s = round(wavenumbers[s])
    wn_e = round(wavenumbers[e - 1])
    FULL_SEGMENTS[f'segment_{i+1}'] = (wn_s, wn_e)

print("\n--- FINAL SEGMENTS (auto-calibrated by variance拐点) ---")
print("FULL_SEGMENTS = {")
for name, (start, end) in FULL_SEGMENTS.items():
    print(f"    '{name}': ({start}, {end}),")
print("}")

# -----------------------------
# 6. Fourier analysis with unified y-scale
# -----------------------------
print("\n" + "="*60)
print("Fourier analysis on each auto-split segment...")
print("="*60)

# Collect all y-values for unified scaling
all_y_values = []
reconstruction_data = []

for i, (start_idx, end_idx) in enumerate(segment_indices):
    sub_wn = wavenumbers[start_idx:end_idx]
    sub_avg = np.mean(all_spectra[:, start_idx:end_idx], axis=0)
    N = len(sub_wn)
    
    if N < 10:
        sine = sub_avg
    else:
        dx = np.mean(np.diff(sub_wn))
        signal = sub_avg - np.mean(sub_avg)
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, d=dx)
        
        if len(fft_vals) > 1:
            dom_idx = np.argmax(np.abs(fft_vals[1:])) + 1
        else:
            dom_idx = 0
            
        f_dom = freqs[dom_idx]
        amp = np.abs(fft_vals[dom_idx])
        phase = np.angle(fft_vals[dom_idx])
        sine = amp * np.cos(2 * np.pi * f_dom * sub_wn + phase) + np.mean(sub_avg)
    
    reconstruction_data.append({
        'sub_wn': sub_wn,
        'sub_avg': sub_avg,
        'sine': sine,
        'start': sub_wn[0],
        'end': sub_wn[-1],
        'f_dom': f_dom if N >= 10 else 0
    })
    
    all_y_values.extend(sub_avg)
    all_y_values.extend(sine)

y_min = min(all_y_values)
y_max = max(all_y_values)
y_padding = (y_max - y_min) * 0.05
y_min -= y_padding
y_max += y_padding

# Plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for i, ax in enumerate(axes):
    d = reconstruction_data[i]
    ax.plot(d['sub_wn'], d['sine'], '--', 
            label=f'Dominant Sine (f={d["f_dom"]:.5f})', color='tab:red')
    ax.set_title(f"Segment {i+1}: [{d['start']:.1f}, {d['end']:.1f}] cm⁻¹", fontsize=12)
    ax.set_ylabel("Intensity")
    ax.set_ylim(y_min, y_max)
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Raman Shift (cm⁻¹)")
plt.tight_layout()
plt.savefig("raman_auto_split_segments_fourier.png", dpi=150)
plt.show()

print("\n✅ Analysis complete with auto-calibrated split points.")
print("Plot saved as 'raman_auto_split_segments_fourier.png'")