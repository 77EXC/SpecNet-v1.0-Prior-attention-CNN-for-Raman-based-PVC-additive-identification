import json
from sklearn.model_selection import train_test_split
import os

# ==============================================================================
#                              *** 配置区 ***
# ==============================================================================
# 1. 输入的已处理好的JSON文件
INPUT_JSON = "raman_dataset_processed_interpolated.json"

# 2. 输出的训练集和验证集文件名
OUTPUT_TRAIN_JSON = "train_dataset.json"
OUTPUT_VAL_JSON = "validation_dataset.json"

# 3. 验证集所占的比例 (10%)
VALIDATION_SIZE = 0.2

# 4. 随机种子，用于确保每次划分的结果都完全相同，保证实验的可复现性
RANDOM_STATE = 42
# ==============================================================================

def main():
    """
    主函数，执行数据加载、划分和保存。
    """
    # --- 1. 加载数据 ---
    if not os.path.exists(INPUT_JSON):
        print(f"错误：找不到输入文件 '{INPUT_JSON}'。请确保脚本和该文件在同一目录下。")
        return
        
    print(f"正在从 '{INPUT_JSON}' 加载数据...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    print(f"加载成功！共找到 {len(all_data)} 个样本。")
    
    # 检查数据量是否足够
    if len(all_data) < 2:
        print("错误：数据量太少，无法进行划分。")
        return

    # --- 2. 使用 scikit-learn 进行数据划分 ---
    print(f"\n正在将数据划分为 {1-VALIDATION_SIZE:.0%} 训练集和 {VALIDATION_SIZE:.0%} 验证集...")
    
    # train_test_split 是一个非常强大的工具，可以轻松完成随机、分层的划分
    # 我们在这里直接对包含完整样本信息（ID, 光谱, 组分等）的列表进行划分
    train_data, validation_data = train_test_split(
        all_data,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True  # 在划分前先随机打乱数据，这是个好习惯
    )
    
    print("划分完成。")
    print(f"  - 训练集样本数: {len(train_data)}")
    print(f"  - 验证集样本数: {len(validation_data)}")

    # --- 3. 保存划分后的数据到新文件 ---
    try:
        print(f"\n正在保存训练集到 '{OUTPUT_TRAIN_JSON}'...")
        with open(OUTPUT_TRAIN_JSON, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        print("训练集保存成功。")

        print(f"\n正在保存验证集到 '{OUTPUT_VAL_JSON}'...")
        with open(OUTPUT_VAL_JSON, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, ensure_ascii=False, indent=4)
        print("验证集保存成功。")
    except Exception as e:
        print(f"\n❌ 保存文件时发生错误: {e}")

if __name__ == "__main__":
    main()