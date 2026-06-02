# train.py

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import os
import re
import matplotlib.pyplot as plt
import pickle # ✅ 确保导入 pickle 库用于保存依赖项

# ==============================================================================
# 1. 数据集类 (包含数据增强)
# ==============================================================================
class RamanSpectraDataset(Dataset):
    """
    自定义PyTorch数据集类，包含丰富的训练时数据增强。
    """
    def __init__(self, spectra, labels, contents, is_train=False):
        self.spectra = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.contents = torch.tensor(contents, dtype=torch.float32)
        self.is_train = is_train

    def __len__(self): return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx].clone()
        
        if self.is_train:
            # 随机高斯噪声
            noise_std = np.random.uniform(0.005, 0.03)
            spectrum += torch.randn(spectrum.shape) * noise_std
            
            # 随机强度缩放
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            spectrum *= scale

            # 随机基线偏移
            base_shift_magnitude = np.random.uniform(-0.05, 0.05)
            if spectrum.shape[-1] > 1:
                linear_base = torch.linspace(base_shift_magnitude, -base_shift_magnitude, spectrum.shape[-1])
                spectrum += linear_base.unsqueeze(0)
            
            # 随机波数抖动
            shift_pixels = np.random.randint(-5, 6)
            if shift_pixels != 0:
                spectrum = torch.roll(spectrum, shifts=shift_pixels, dims=-1)
                if shift_pixels > 0:
                    spectrum[:, :shift_pixels] = spectrum[:, shift_pixels]
                else:
                    spectrum[:, shift_pixels:] = spectrum[:, shift_pixels-1]
            
            # 最后再次归一化到0-1
            min_val, max_val = spectrum.min(), spectrum.max()
            if (max_val - min_val) > 1e-6:
                spectrum = (spectrum - min_val) / (max_val - min_val)
            else:
                spectrum -= min_val
                
        return spectrum, self.labels[idx], self.contents[idx]

# ==============================================================================
# 2. 标签清洗与归一化函数
# ==============================================================================
def clean_and_normalize_labels(component_dicts):
    cleaned_component_dicts = []
    for comp_dict in component_dicts:
        new_dict = {}
        for key, value in comp_dict.items():
            clean_key = str(key).strip().upper()
            try: numeric_value = float(value)
            except (ValueError, TypeError): numeric_value = 0.0
            
            if 'TIO' in clean_key: clean_key = 'TIO2'
            elif 'MGCO' in clean_key: clean_key = 'MGCO3'
            
            new_dict[clean_key] = new_dict.get(clean_key, 0.0) + numeric_value
        cleaned_component_dicts.append(new_dict)
    return cleaned_component_dicts

# ==============================================================================
# 3. 可配置的多分支CNN模型
# ==============================================================================
class ConfigurableMultiBranchCNN(nn.Module):
    def __init__(self, band_indices, num_classes, config=None):
        super(ConfigurableMultiBranchCNN, self).__init__()
        if config is None: config = self.get_default_config()
        
        self.band_indices = band_indices
        total_fusion_dim = 0
        
        self.branch_fingerprint, dim = self._create_branch(config['fingerprint']); total_fusion_dim += dim
        self.branch_mid_freq, dim = self._create_branch(config['mid_freq']); total_fusion_dim += dim
        self.branch_high_freq, dim = self._create_branch(config['high_freq']); total_fusion_dim += dim
        self.branch_global, dim = self._create_branch(config['global']); total_fusion_dim += dim
        
        fusion_cfg = config['fusion_head']
        self.fusion_head = nn.Sequential(
            nn.Linear(total_fusion_dim, fusion_cfg['hidden_dim']),
            nn.BatchNorm1d(fusion_cfg['hidden_dim']), nn.ReLU(),
            nn.Dropout(fusion_cfg['dropout'])
        )
        self.classification_head = nn.Linear(fusion_cfg['hidden_dim'], num_classes)
        self.content_head = nn.Sequential(nn.Linear(fusion_cfg['hidden_dim'], num_classes), nn.Sigmoid())

    def _create_branch(self, branch_config):
        layers, in_channels = [], 1
        for out_channels, kernel_size, pool_size in branch_config:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False),
                nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)
            ])
            if pool_size > 1: layers.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
        return nn.Sequential(*layers), in_channels

    def forward(self, x):
        x_fingerprint = x[:, :, self.band_indices['fingerprint'][0]:self.band_indices['fingerprint'][1]]
        x_mid = x[:, :, self.band_indices['mid_freq'][0]:self.band_indices['mid_freq'][1]]
        x_high = x[:, :, self.band_indices['high_freq'][0]:self.band_indices['high_freq'][1]]
        
        f_fingerprint = self.branch_fingerprint(x_fingerprint)
        f_mid = self.branch_mid_freq(x_mid)
        f_high = self.branch_high_freq(x_high)
        f_global = self.branch_global(x)
        
        fusion = torch.cat([f_fingerprint, f_mid, f_high, f_global], dim=1)
        shared_features = self.fusion_head(fusion)
        cls_outputs = self.classification_head(shared_features)
        content_outputs = self.content_head(shared_features)
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
# 4. 损失函数
# ==============================================================================
class DynamicMatchingLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(DynamicMatchingLoss, self).__init__()
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.L1Loss()
        self.alpha, self.beta = alpha, beta
    
    def forward(self, pred_cls, pred_content, true_labels, true_content):
        pos_mask = (true_labels > 0).float()
        loss_cls = self.cls_loss_fn(pred_cls, true_labels)
        loss_reg = self.reg_loss_fn(pred_content[pos_mask > 0], true_content[pos_mask > 0]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=pred_cls.device)
        return self.alpha * loss_cls + self.beta * loss_reg, loss_cls, loss_reg

# ==============================================================================
# 5. 训练与评估函数
# ==============================================================================
def train_model(model, train_loader, val_loader, epochs, lr, device, save_path, patience):
    criterion = DynamicMatchingLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4) # 增加权重衰减以对抗过拟合
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for spectra, labels, contents in train_loader:
            spectra, labels, contents = spectra.to(device), labels.to(device), contents.to(device)
            optimizer.zero_grad()
            cls_out, content_out = model(spectra)
            loss, _, _ = criterion(cls_out, content_out, labels, contents)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for spectra, labels, contents in val_loader:
                spectra, labels, contents = spectra.to(device), labels.to(device), contents.to(device)
                cls_out, content_out = model(spectra)
                loss, _, _ = criterion(cls_out, content_out, labels, contents)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Validation loss decreased. Saving best model to '{save_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
        scheduler.step(avg_val_loss)
            
    return history

def evaluate_and_predict(model, data_loader, mlb, content_scaler, device, data_name="验证集", threshold=0.5):
    print("\n" + "="*50 + f"\n开始在【{data_name}】上进行最终评估...\n" + "="*50)
    model.eval()
    all_labels, all_preds_binary = [], []
    with torch.no_grad():
        for spectra, labels, _ in data_loader:
            spectra = spectra.to(device)
            cls_out, _ = model(spectra)
            all_labels.extend(labels.cpu().numpy())
            all_preds_binary.extend(torch.sigmoid(cls_out).cpu().numpy() > threshold)
            
    all_labels, all_preds_binary = np.array(all_labels), np.array(all_preds_binary)
    print(f"子集准确率 (Exact Match Ratio): {accuracy_score(all_labels, all_preds_binary):.4f}")
    print(f"平均精确率 (Avg Precision):     {precision_score(all_labels, all_preds_binary, average='samples', zero_division=0):.4f}")
    print(f"平均召回率 (Avg Recall):        {recall_score(all_labels, all_preds_binary, average='samples', zero_division=0):.4f}")

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数量: {total_params:,}")

# ==============================================================================
# 主程序 (Main Program)
# ==============================================================================
if __name__ == "__main__":
    # --- 配置区 ---
    TRAIN_JSON = "train_dataset.json"
    VAL_JSON = "validation_dataset.json"
    MODEL_SAVE_PATH = "best_configurable_cnn.pt"
    LOSS_CURVE_PATH = "loss_curves_configurable_cnn.png"
    PREPROCESSOR_SAVE_PATH = "inference_dependencies.pkl" # ✅ 保存依赖项的文件名
    
    BATCH_SIZE = 32
    EPOCHS = 500
    LEARNING_RATE = 0.001
    PATIENCE = 50 # 增加早停的耐心
    WAVE_BANDS = {'fingerprint': (0, 1750), 'mid_freq': (1750, 2750), 'high_freq': (2750, 3500)}

    # --- 1. 数据加载与准备 ---
    if not (os.path.exists(TRAIN_JSON) and os.path.exists(VAL_JSON)):
        print(f"错误: 确保 '{TRAIN_JSON}' 和 '{VAL_JSON}' 文件存在。"); exit()
    
    print("正在加载和准备数据...");
    with open(TRAIN_JSON, 'r', encoding='utf-8') as f: train_data_raw = json.load(f)
    with open(VAL_JSON, 'r', encoding='utf-8') as f: val_data_raw = json.load(f)

    train_components = clean_and_normalize_labels([s['components'] for s in train_data_raw])
    val_components = clean_and_normalize_labels([s['components'] for s in val_data_raw])

    mlb = MultiLabelBinarizer()
    y_train_encoded = mlb.fit_transform([list(d.keys()) for d in train_components])
    y_val_encoded = mlb.transform([list(d.keys()) for d in val_components])
    num_classes = len(mlb.classes_)
    print(f"\n清洗后，在训练集上检测到 {num_classes} 个唯一的组分: {list(mlb.classes_)}")

    train_contents = np.array([[d.get(cls, 0.0) for cls in mlb.classes_] for d in train_components])
    val_contents = np.array([[d.get(cls, 0.0) for cls in mlb.classes_] for d in val_components])
    
    content_scaler = MinMaxScaler()
    y_train_content_scaled = content_scaler.fit_transform(train_contents)
    y_val_content_scaled = content_scaler.transform(val_contents)

    train_spectra = np.array([s['raman_spectrum']['intensity_processed'] for s in train_data_raw])
    val_spectra = np.array([s['raman_spectrum']['intensity_processed'] for s in val_data_raw])
    
    train_dataset = RamanSpectraDataset(train_spectra, y_train_encoded, y_train_content_scaled, is_train=True)
    val_dataset = RamanSpectraDataset(val_spectra, y_val_encoded, y_val_content_scaled, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. 计算波数段索引 ---
    wavenumbers = train_data_raw[0]['raman_spectrum']['wavenumber_processed']
    wavenumbers_np = np.array(wavenumbers)
    band_indices = {}
    for name, (start, end) in WAVE_BANDS.items():
        start_idx = np.argmin(np.abs(wavenumbers_np - start))
        end_idx = np.argmin(np.abs(wavenumbers_np - end))
        print(f"波段 '{name}' ({start}-{end} cm⁻¹): 索引范围 [{start_idx}:{end_idx}]")
        band_indices[name] = (start_idx, end_idx)

    # --- 3. 构建并训练模型 ---
    chosen_config = ConfigurableMultiBranchCNN.get_default_config()
    model = ConfigurableMultiBranchCNN(band_indices=band_indices, num_classes=num_classes, config=chosen_config)
    print("\n--- 使用默认模型配置 ---")
    print_model_parameters(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n将使用 '{device}' 设备进行训练。")
    
    history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=device, save_path=MODEL_SAVE_PATH, patience=PATIENCE)
    
    # --- 4. 绘制损失曲线 ---
    print("\n训练完成。")
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.yscale('log')
    plt.savefig(LOSS_CURVE_PATH)
    print(f"损失曲线图已保存至 '{LOSS_CURVE_PATH}'")

    # --- 5. ✅ 保存推理所需的所有依赖项 ---
    print(f"\n正在将推理所需的依赖项保存到 '{PREPROCESSOR_SAVE_PATH}'...")
    inference_dependencies = {
        'mlb': mlb,
        'content_scaler': content_scaler,
        'band_indices': band_indices,
        'model_config': chosen_config,
        'num_classes': num_classes
    }
    with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
        pickle.dump(inference_dependencies, f)
    print("依赖项保存成功！")
    
    # --- 6. 加载最佳模型并进行最终评估 ---
    print(f"\n正在从 '{MODEL_SAVE_PATH}' 加载性能最佳的模型权重进行最终评估...")
    best_model = ConfigurableMultiBranchCNN(band_indices=band_indices, num_classes=num_classes, config=chosen_config)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    best_model.to(device)
    evaluate_and_predict(best_model, val_loader, mlb, content_scaler, device, data_name="验证集 (使用最佳权重)")