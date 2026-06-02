import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pybaselines.whittaker import arpls
import random

def preprocess_spectrum(intensity_raw):
    """
    对单条拉曼光谱数据进行完整的预处理流程。
    """
    intensity_np = np.array(intensity_raw)
    
    # 1. 基线校正 (arpls)
    baseline, params = arpls(intensity_np, lam=1e6)
    intensity_baseline_corrected = intensity_np - baseline
    
    # 2. 平滑去噪 (Savitzky-Golay滤波器)
    if len(intensity_baseline_corrected) > 11:
        intensity_smoothed = savgol_filter(intensity_baseline_corrected, window_length=11, polyorder=3)
    else:
        intensity_smoothed = intensity_baseline_corrected
        
    # 3. 归一化 (Min-Max Normalization)
    min_val = np.min(intensity_smoothed)
    max_val = np.max(intensity_smoothed)
    if max_val - min_val > 1e-6:
        intensity_normalized = (intensity_smoothed - min_val) / (max_val - min_val)
    else:
        intensity_normalized = intensity_smoothed - min_val
        
    return intensity_normalized


def main():
    """
    主函数，执行数据加载、插值、批量处理、保存和可视化。
    """
    # --- 1. 定义文件路径和关键词 ---
    input_json_path = "raman_dataset_cleaned_v2.json"
    output_json_path = "raman_dataset_processed_interpolated.json"
    HOLD_OUT_KEYWORD = "泛化集" # 定义用于识别泛化集的关键词

    # --- 2. 加载JSON数据 ---
    if not os.path.exists(input_json_path):
        print(f"Error: Input file '{input_json_path}' not found.")
        return
        
    print(f"Loading data from '{input_json_path}'...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        all_samples_data = json.load(f)
    print(f"Load successful! Found {len(all_samples_data)} total samples.")

    # --- 3. 创建一个统一的、高分辨率的公共波数网格 ---
    # 注意：我们仍然在所有数据上创建网格，以保证波数轴的一致性
    print("\nCreating a common wavenumber grid for all samples...")
    global_min_wn = float('inf')
    global_max_wn = float('-inf')
    valid_samples = [s for s in all_samples_data if s["raman_spectrum"]["wavenumber"]]
    
    if not valid_samples:
        print("Error: No valid samples with wavenumber data found.")
        return

    for sample in valid_samples:
        wn = sample["raman_spectrum"]["wavenumber"]
        global_min_wn = min(global_min_wn, wn[0])
        global_max_wn = max(global_max_wn, wn[-1])

    num_points = len(valid_samples[0]["raman_spectrum"]["wavenumber"])
    common_wavenumber_grid = np.linspace(global_min_wn, global_max_wn, num_points)
    print(f"Common grid created: Range {global_min_wn:.2f} - {global_max_wn:.2f} cm⁻¹, Points: {num_points}")

    # --- 4. 批量处理所有样本 (包含插值)，并跳过泛化集 ---
    print("\nStarting batch interpolation and processing...")
    print(f"Samples containing '{HOLD_OUT_KEYWORD}' in their ID will be skipped.")
    
    processed_samples_list = []
    skipped_count = 0
    
    for i, sample in enumerate(all_samples_data):
        
        # ✅ *** 核心修改: 检查并跳过泛化集样本 ***
        if HOLD_OUT_KEYWORD in sample['sample_id']:
            skipped_count += 1
            continue # 跳过当前循环的剩余部分

        # 打印进度 (只对处理的样本计数)
        processed_count = i + 1 - skipped_count
        total_to_process = len(all_samples_data) - skipped_count
        if processed_count % 100 == 0 or processed_count == total_to_process:
            print(f"  Processing... {processed_count}/{total_to_process}")
            
        original_wavenumber = sample["raman_spectrum"]["wavenumber"]
        original_intensity = sample["raman_spectrum"]["intensity"]
        
        if not original_wavenumber or not original_intensity:
            print(f"Warning: Sample '{sample['sample_id']}' has empty spectral data. Skipping.")
            continue
            
        interpolated_intensity = np.interp(common_wavenumber_grid, original_wavenumber, original_intensity)
        processed_intensity = preprocess_spectrum(interpolated_intensity)
        
        sample["raman_spectrum"]["wavenumber_processed"] = common_wavenumber_grid.tolist()
        sample["raman_spectrum"]["intensity_processed"] = processed_intensity.tolist()
        
        processed_samples_list.append(sample)

    print(f"\n✅ Batch processing complete!")
    print(f"Total samples processed and saved: {len(processed_samples_list)}")
    print(f"Total samples skipped (hold-out set): {skipped_count}")


    # --- 5. 保存处理后的数据到新JSON文件 ---
    print(f"\nSaving processed data to '{output_json_path}'...")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_samples_list, f, ensure_ascii=False, indent=4)
        print("Save successful!")
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")

    # --- 6. 可视化几个样本以验证结果 (使用纯英文标签) ---
    print("\nGenerating visualization to compare results...")
    # 从已处理的列表中随机抽样
    num_samples_to_plot = min(len(processed_samples_list), 3)
    
    if num_samples_to_plot > 0:
        sample_indices = random.sample(range(len(processed_samples_list)), num_samples_to_plot)
        
        fig, axs = plt.subplots(num_samples_to_plot, 1, figsize=(12, 5 * num_samples_to_plot), squeeze=False)

        for i, idx in enumerate(sample_indices):
            sample_to_plot = processed_samples_list[idx]
            original_wn = sample_to_plot["raman_spectrum"]["wavenumber"]
            original_int = sample_to_plot["raman_spectrum"]["intensity"]
            processed_wn = sample_to_plot["raman_spectrum"]["wavenumber_processed"]
            processed_int = sample_to_plot["raman_spectrum"]["intensity_processed"]
            
            ax = axs[i, 0]
            ax.plot(original_wn, original_int, label='Original Data', color='blue', alpha=0.6)
            
            ax2 = ax.twinx()
            ax2.plot(processed_wn, processed_int, label='Processed Data', color='red')

            ax.set_title(f"Sample ID: {sample_to_plot['sample_id']}")
            ax.set_xlabel('Raman Shift (cm⁻¹)')
            ax.set_ylabel('Original Intensity', color='blue')
            ax2.set_ylabel('Processed Intensity (Normalized)', color='red')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        output_figure_path = "comparison_plot.png"
        plt.savefig(output_figure_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to file: {output_figure_path}")

if __name__ == "__main__":
    main()