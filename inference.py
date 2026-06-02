# inference.py

import torch
import json
import numpy as np
import pickle
import argparse
import torch.nn as nn # 必须导入nn以定义模型类

# ==============================================================================
# 1. 复制模型定义
# 必须将与训练时完全相同的模型类定义复制到这里。
# ==============================================================================
class ConfigurableMultiBranchCNN(nn.Module):
    def __init__(self, band_indices, num_classes, config=None):
        super(ConfigurableMultiBranchCNN, self).__init__()
        if config is None:
            config = self.get_default_config()
        self.band_indices = band_indices
        total_fusion_dim = 0
        self.branch_fingerprint, fusion_dim = self._create_branch(config['fingerprint']); total_fusion_dim += fusion_dim
        self.branch_mid_freq, fusion_dim = self._create_branch(config['mid_freq']); total_fusion_dim += fusion_dim
        self.branch_high_freq, fusion_dim = self._create_branch(config['high_freq']); total_fusion_dim += fusion_dim
        self.branch_global, fusion_dim = self._create_branch(config['global']); total_fusion_dim += fusion_dim
        fusion_hidden_dim = config['fusion_head']['hidden_dim']; dropout_rate = config['fusion_head']['dropout']
        self.fusion_head = nn.Sequential(nn.Linear(total_fusion_dim, fusion_hidden_dim), nn.BatchNorm1d(fusion_hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.classification_head = nn.Linear(fusion_hidden_dim, num_classes)
        self.content_head = nn.Sequential(nn.Linear(fusion_hidden_dim, num_classes), nn.Sigmoid())
    def _create_branch(self, branch_config):
        layers = []; in_channels = 1
        for layer_params in branch_config:
            out_channels, kernel_size, pool_size = layer_params
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False))
            layers.append(nn.BatchNorm1d(out_channels)); layers.append(nn.ReLU(inplace=True))
            if pool_size > 1: layers.append(nn.MaxPool1d(kernel_size=pool_size))
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool1d(1)); layers.append(nn.Flatten())
        return nn.Sequential(*layers), in_channels
    def forward(self, x):
        x_fingerprint = x[:, :, self.band_indices['fingerprint'][0]:self.band_indices['fingerprint'][1]]
        x_mid = x[:, :, self.band_indices['mid_freq'][0]:self.band_indices['mid_freq'][1]]
        x_high = x[:, :, self.band_indices['high_freq'][0]:self.band_indices['high_freq'][1]]
        f_fingerprint = self.branch_fingerprint(x_fingerprint); f_mid = self.branch_mid_freq(x_mid)
        f_high = self.branch_high_freq(x_high); f_global = self.branch_global(x)
        fusion = torch.cat([f_fingerprint, f_mid, f_high, f_global], dim=1); shared_features = self.fusion_head(fusion)
        cls_outputs = self.classification_head(shared_features); content_outputs = self.content_head(shared_features)
        return cls_outputs, content_outputs
    @staticmethod
    def get_default_config():
        return {
            'fingerprint': [ (64, 3, 1), (128, 3, 2), (256, 3, 2) ],
            'mid_freq':    [ (64, 7, 4), (128, 7, 1) ],
            'high_freq':   [ (64, 5, 4), (128, 5, 1) ],
            'global':      [ (64, 5, 2), (128, 5, 2), (256, 5, 2) ],
            'fusion_head': { 'hidden_dim': 512, 'dropout': 0.5 }
        }

# ==============================================================================
# 2. 推理主函数
# ==============================================================================
def predict_single_sample(sample_id, data_path, model_path, preprocessor_path, threshold=0.5):
    """
    对单个样本进行加载、预处理、预测和后处理。
    """
    # --- 1. 加载所有依赖项 ---
    print(f"--- 正在加载模型及依赖项 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        with open(preprocessor_path, 'rb') as f:
            dependencies = pickle.load(f)
        mlb = dependencies['mlb']
        content_scaler = dependencies['content_scaler']
        band_indices = dependencies['band_indices']
        model_config = dependencies['model_config']
        num_classes = len(mlb.classes_)
    except FileNotFoundError:
        print(f"错误: 找不到依赖项文件 '{preprocessor_path}'。请先运行训练脚本生成它。")
        return

    # --- 2. 加载模型权重 ---
    model = ConfigurableMultiBranchCNN(
        band_indices=band_indices,
        num_classes=num_classes,
        config=model_config
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # 切换到评估模式
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"错误: 找不到模型权重文件 '{model_path}'。")
        return
    
    # --- 3. 查找并加载样本数据 ---
    print(f"\n--- 正在数据文件 '{data_path}' 中查找 Sample ID: {sample_id} ---")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{data_path}'。")
        return

    target_sample = None
    for sample in all_data:
        if sample.get('sample_id') == sample_id:
            target_sample = sample
            break
    
    if target_sample is None:
        print(f"错误: 在数据文件中未找到 Sample ID '{sample_id}'。")
        return
    
    print("成功找到样本！")
    
    # --- 4. 预处理输入光谱 ---
    spectrum_intensity = np.array(target_sample['raman_spectrum']['intensity_processed'], dtype=np.float32)
    spectrum_tensor = torch.tensor(spectrum_intensity).unsqueeze(0).unsqueeze(0) # -> [1, 1, seq_len]
    spectrum_tensor = spectrum_tensor.to(device)

    # --- 5. 执行推理 ---
    print("\n--- 开始执行推理 ---")
    with torch.no_grad():
        cls_logits, content_output_scaled = model(spectrum_tensor)
    
    # --- 6. 后处理输出 ---
    # 将输出从GPU移至CPU并转为Numpy
    cls_logits_np = cls_logits.cpu().numpy().flatten()
    content_output_scaled_np = content_output_scaled.cpu().numpy().flatten()
    
    # 计算概率
    cls_probs = 1 / (1 + np.exp(-cls_logits_np))
    
    # 根据阈值确定预测的组分
    predicted_indices = np.where(cls_probs > threshold)[0]
    predicted_components = mlb.classes_[predicted_indices]

    # 构建最终的预测报告
    prediction_report = {}
    if len(predicted_components) > 0:
        # 只对预测为存在的组分进行含量逆变换
        predicted_contents_scaled = content_output_scaled_np[predicted_indices]
        
        # 准备一个完整的向量以进行逆变换
        full_content_vector = np.zeros_like(content_output_scaled_np)
        full_content_vector[predicted_indices] = predicted_contents_scaled
        
        # 执行逆变换
        predicted_contents_unscaled = content_scaler.inverse_transform(full_content_vector.reshape(1, -1)).flatten()

        for i, comp_name in enumerate(predicted_components):
            prediction_report[comp_name] = round(float(predicted_contents_unscaled[predicted_indices[i]]), 4)

    # --- 7. 打印结果 ---
    print("\n" + "="*50)
    print(f"✅ 推理结果 for Sample ID: {sample_id}")
    print("="*50)
    print(f"真实组分: {list(target_sample['components'].keys())}")
    print("\n--- 模型预测报告 (组分: 预测含量) ---")
    if not prediction_report:
        print("模型未预测到任何高于阈值的组分。")
    else:
        # 使用json模块美化打印
        print(json.dumps(prediction_report, indent=4))
    print("="*50)


# ==============================================================================
# 3. 命令行参数解析
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对单个拉曼光谱样本进行推理。")
    parser.add_argument("sample_id", type=str, help="要进行预测的样本ID (例如: 'sample_1')。")
    parser.add_argument("--data_path", type=str, default="validation_dataset.json", help="包含样本数据的JSON文件路径。")
    parser.add_argument("--model_path", type=str, default="best_configurable_cnn.pt", help="已训练的模型权重文件 (.pt) 的路径。")
    parser.add_argument("--preprocessor_path", type=str, default="inference_dependencies.pkl", help="包含mlb和scaler的依赖项文件 (.pkl) 的路径。")
    parser.add_argument("--threshold", type=float, default=0.5, help="用于判断组分是否存在的概率阈值。")
    
    args = parser.parse_args()
    
    predict_single_sample(
        sample_id=args.sample_id,
        data_path=args.data_path,
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
        threshold=args.threshold
    )