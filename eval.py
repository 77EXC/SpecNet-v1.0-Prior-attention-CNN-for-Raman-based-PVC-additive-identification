# eval.py - 修正版：与 train_optimized_final_dropout.py 完全对齐

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import pickle
import os

# ==============================================================================
# 1. 数据集类（仅用于推理）
# ==============================================================================
class RamanSpectraDataset(Dataset):
    def __init__(self, spectra, labels, contents):
        self.spectra = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.contents = torch.tensor(contents, dtype=torch.float32)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx], self.contents[idx]

# ==============================================================================
# 2. 标签清洗函数（必须与训练一致）
# ==============================================================================
def clean_and_normalize_labels(component_dicts):
    cleaned_component_dicts = []
    for comp_dict in component_dicts:
        new_dict = {}
        for key, value in comp_dict.items():
            clean_key = str(key).strip().upper()
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                numeric_value = 0.0
            
            if 'TIO' in clean_key:
                clean_key = 'TIO2'
            elif 'MGCO' in clean_key:
                clean_key = 'MGCO3'
            
            new_dict[clean_key] = new_dict.get(clean_key, 0.0) + numeric_value
        cleaned_component_dicts.append(new_dict)
    return cleaned_component_dicts

# ==============================================================================
# 3. 模型定义（必须与训练脚本完全一致！）
# ==============================================================================
class BasicBlock1D(nn.Module):
    """一维 ResNet 基本残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out += identity; out = self.relu(out)
        return out

class ResNetLikeMultiBranchCNN(nn.Module):
    """
    使用残差块优化的多分支CNN模型。
    配置格式: [(num_blocks, out_channels, stride), ...]
    """
    def __init__(self, band_indices, num_classes, config=None):
        super(ResNetLikeMultiBranchCNN, self).__init__()
        if config is None: config = self.get_default_config()
        
        self.band_indices = band_indices
        total_fusion_dim = 0
        
        # 初始层：将 [B, 1, L] 预处理成 [B, 64, L']
        self.init_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # 创建分支 (所有分支都从 64 个通道开始)
        self.branch_fingerprint, dim = self._create_branch_with_resblocks(config['fingerprint'], in_channels=64)
        total_fusion_dim += dim
        self.branch_mid_freq, dim = self._create_branch_with_resblocks(config['mid_freq'], in_channels=64)
        total_fusion_dim += dim
        self.branch_high_freq, dim = self._create_branch_with_resblocks(config['high_freq'], in_channels=64)
        total_fusion_dim += dim
        self.branch_global, dim = self._create_global_branch(config['global'])
        total_fusion_dim += dim
        
        # 融合与输出头
        fusion_cfg = config['fusion_head']
        self.fusion_head = nn.Sequential(
            nn.Linear(total_fusion_dim, fusion_cfg['hidden_dim']),
            nn.BatchNorm1d(fusion_cfg['hidden_dim']), nn.ReLU(),
            nn.Dropout(fusion_cfg['dropout'])
        )
        self.classification_head = nn.Linear(fusion_cfg['hidden_dim'], num_classes)
        self.content_head = nn.Sequential(nn.Linear(fusion_cfg['hidden_dim'], num_classes), nn.Sigmoid())

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        stages = []
        stages.append(BasicBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            stages.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*stages)

    def _create_branch_with_resblocks(self, branch_config, in_channels):
        layers = []
        current_channels = in_channels
        for num_blocks, out_channels, stride in branch_config:
            layers.append(self._make_stage(current_channels, out_channels, num_blocks, stride))
            current_channels = out_channels
        layers.extend([
            nn.Dropout(p=0.2),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten()
        ])
        return nn.Sequential(*layers), current_channels
    
    def _create_global_branch(self, global_config):
        in_channels = 64
        current_channels = in_channels
        layers = []
        for num_blocks, out_channels, stride in global_config:
            layers.append(self._make_stage(current_channels, out_channels, num_blocks, stride))
            current_channels = out_channels
        layers.extend([
            nn.Dropout(p=0.2),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten()
        ])
        return nn.Sequential(*layers), current_channels

    def forward(self, x_global):
        x_features = self.init_conv(x_global) 

        # 使用与 deps['band_indices_feat'] 一致的键名：'fingerprint', 'mid_freq', 'high_freq'
        x_fingerprint = x_features[:, :, self.band_indices['fingerprint'][0]:self.band_indices['fingerprint'][1]]
        x_mid = x_features[:, :, self.band_indices['mid_freq'][0]:self.band_indices['mid_freq'][1]]
        x_high = x_features[:, :, self.band_indices['high_freq'][0]:self.band_indices['high_freq'][1]]
        
        f_fingerprint = self.branch_fingerprint(x_fingerprint)
        f_mid = self.branch_mid_freq(x_mid)
        f_high = self.branch_high_freq(x_high)
        f_global = self.branch_global(x_features) 
        
        fusion = torch.cat([f_fingerprint, f_mid, f_high, f_global], dim=1)
        shared_features = self.fusion_head(fusion)
        cls_outputs = self.classification_head(shared_features)
        content_outputs = self.content_head(shared_features)
        return cls_outputs, content_outputs

    @staticmethod
    def get_default_config():
        return {
            'fingerprint': [ (2, 128, 2), (2, 256, 2) ],
            'mid_freq': [ (2, 128, 2) ],
            'high_freq': [ (2, 128, 1) ],
            'global': [ (2, 128, 2), (2, 256, 2), (2, 512, 2) ],
            'fusion_head': { 'hidden_dim': 1024, 'dropout': 0.5 } 
        }

# ==============================================================================
# 4. 评估函数
# ==============================================================================
def evaluate_model(model, data_loader, mlb, content_scaler, device, threshold=0.5):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_true_contents = []
    all_pred_contents = []

    with torch.no_grad():
        for spectra, labels, contents in data_loader:
            spectra = spectra.to(device)
            cls_out, content_out = model(spectra)
            
            probs = torch.sigmoid(cls_out).cpu().numpy()
            preds_binary = (probs > threshold).astype(int)
            
            all_true_labels.append(labels.cpu().numpy())
            all_pred_labels.append(preds_binary)
            
            pred_contents_scaled = content_out.cpu().numpy()
            pred_contents = content_scaler.inverse_transform(pred_contents_scaled)
            true_contents = content_scaler.inverse_transform(contents.cpu().numpy())
            
            all_true_contents.append(true_contents)
            all_pred_contents.append(pred_contents)

    all_true_labels = np.vstack(all_true_labels)
    all_pred_labels = np.vstack(all_pred_labels)
    all_true_contents = np.vstack(all_true_contents)
    all_pred_contents = np.vstack(all_pred_contents)

    # 分类指标
    exact_match = accuracy_score(all_true_labels, all_pred_labels)
    avg_precision = precision_score(all_true_labels, all_pred_labels, average='samples', zero_division=0)
    avg_recall = recall_score(all_true_labels, all_pred_labels, average='samples', zero_division=0)
    avg_f1 = f1_score(all_true_labels, all_pred_labels, average='samples', zero_division=0)

    # 回归指标
    mae = mean_absolute_error(all_true_contents, all_pred_contents)

    print("="*60)
    print("【最终评估结果】")
    print("="*60)
    print(f"子集准确率 (Exact Match Ratio): {exact_match:.4f}")
    print(f"平均精确率 (Precision):         {avg_precision:.4f}")
    print(f"平均召回率 (Recall):            {avg_recall:.4f}")
    print(f"平均 F1 分数:                   {avg_f1:.4f}")
    print(f"组分含量 MAE (回归误差):        {mae:.4f}")
    print("="*60)

    return {
        'exact_match': exact_match,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'mae': mae
    }

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    TEST_JSON = "validation_dataset.json"
    MODEL_PATH = "best_resnet_multibranch_v2.pt"
    DEPENDENCIES_PATH = "inference_dependencies_resnet_multi_v2.pkl"  # 注意文件名要匹配！

    assert os.path.exists(TEST_JSON), f"测试文件 {TEST_JSON} 不存在！"
    assert os.path.exists(MODEL_PATH), f"模型文件 {MODEL_PATH} 不存在！"
    assert os.path.exists(DEPENDENCIES_PATH), f"依赖文件 {DEPENDENCIES_PATH} 不存在！"

    # --- 1. 加载依赖项 ---
    print("正在加载推理依赖项...")
    with open(DEPENDENCIES_PATH, 'rb') as f:
        deps = pickle.load(f)
    
    mlb = deps['mlb']
    content_scaler = deps['content_scaler']
    # ⚠️ 关键：使用特征图上的索引（不是原始光谱索引）
    band_indices_feat = deps['band_indices_feat']  # 这个才是模型 forward 需要的！
    model_config = deps['model_config']
    num_classes = deps['num_classes']

    # --- 2. 加载测试数据 ---
    print("正在加载测试数据...")
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data_raw = json.load(f)

    test_components = clean_and_normalize_labels([s['components'] for s in test_data_raw])
    y_test_encoded = mlb.transform([list(d.keys()) for d in test_components])
    test_contents = np.array([[d.get(cls, 0.0) for cls in mlb.classes_] for d in test_components])
    y_test_content_scaled = content_scaler.transform(test_contents)
    test_spectra = np.array([s['raman_spectrum']['intensity_processed'] for s in test_data_raw])

    test_dataset = RamanSpectraDataset(test_spectra, y_test_encoded, y_test_content_scaled)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 3. 构建模型并加载权重 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = ResNetLikeMultiBranchCNN(
        band_indices=band_indices_feat,  # 传入的是 {'fingerprint_feat': ..., ...}
        num_classes=num_classes,
        config=model_config
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # --- 4. 执行评估 ---
    results = evaluate_model(model, test_loader, mlb, content_scaler, device, threshold=0.5)

    # --- 5. 保存结果 ---
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=4)
    print("\n评估结果已保存至 'evaluation_results.json'")