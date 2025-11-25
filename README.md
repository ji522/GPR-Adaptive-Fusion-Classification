# GPR 数据集处理与交叉相干系数提取

## 核心目标
本项目针对地质雷达（Ground Penetrating Radar, GPR）原始数据，完成如下标准化流程：
1. 将各类极化信号统一到 224×224 的分辨率并进行六种增强；
2. 计算 VV 与 VH 极化通道之间的交叉相干系数；
3. 组织成可直接用于深度学习训练的数据集（含训练/验证/测试三大划分）。

最终产物是一套仅包含 VV 图像、VH 图像以及 VV-VH 交叉相干矩阵的规范化数据集，可直接交付给其他模型使用。

## 数据处理流程
### 1. 统一尺寸与数据增强
- **脚本**：`unify_dimensions_and_augment_224x224.m`
- **输入**：`merged_complete_dataset_no_augment/` 中的原始复数矩阵（HH/VV/VH 三极化），尺寸 101×28～201×72 不等。
- **操作**：对实部、虚部分别进行双三次插值至 224×224，并生成 6 种增强版本（6倍增强组合）：
  - 旋转：±15°
  - 缩放：0.8x 和 1.2x
  - 水平翻转：50% 概率
  - 增益调整：0.7x 和 1.3x
- **输出**：`unified_augmented_dataset_224x224/`，每个类别 150 组样本（25 个原始样本 × 6 倍增强）。

### 2. 交叉相干系数计算
- **脚本**：`calculate_coherence_matrices_224x224.m`
- **目标**：计算 VV 与 VH 极化之间的交叉相干幅度  
  \[ C_{VV\_VH}(i,j) = \left|S_{VV}(i,j) \cdot \text{conj}\big(S_{VH}(i,j)\big)\right| \]
- **输出变量**：`data1`，尺寸 `[1, 224, 224]` 的单通道矩阵。
- **存储位置**：`unified_augmented_dataset_224x224/<类别>/VV_VH_matrices/`。

> 说明：原始项目中曾计算 HH/VV 等 2×2 相干矩阵。当前版本已按需求仅保留 VV、VH 图像及其交叉相干矩阵，HH 相关内容已移除。

### 3. 神经网络训练数据集重组
- **脚本**：`reorganize_dataset_for_training.py`（Python）或 `reorganize_dataset_for_training.m`（MATLAB）。
- **操作**：
  1. 收集每个类别下的 VV/VH 图像与 VV-VH 矩阵；
  2. 按 70% / 10% / 20% 随机划分训练、测试、验证集；
  3. 重新编号为 `000000_class_<id>_TYPE.ext` 并生成 `labels.csv`；
  4. 输出到 `jsy_dataset_and_model/neural_network_dataset/`。
- **注意**：此步骤**仅处理 VH 图像、VV 图像和 VV_VH 交叉相干矩阵**，不包括 HH 相关数据（HH 图像、HH_matrices、HH_VH_matrices、HH_VV_matrices）。

## 数据集内容
### 1. 增强后的统一尺寸数据集
```
unified_augmented_dataset_224x224/<类别>/
├── HH/                # 150 张 224×224 JPEG，含 6 种增强方式
├── VV/                # 150 张 224×224 JPEG，含 6 种增强方式
├── VH/                # 150 张 224×224 JPEG，含 6 种增强方式
├── HH_matrices/       # 150 个 HH 矩阵 (.mat, 变量 data1)
├── VV_matrices/       # 150 个 VV 矩阵 (.mat, 变量 data1)
├── VH_matrices/       # 150 个 VH 矩阵 (.mat, 变量 data1)
├── HH_VV_matrices/    # 150 个 HH-VV 交叉相干矩阵 (.mat, 变量 data1)
├── HH_VH_matrices/    # 150 个 HH-VH 交叉相干矩阵 (.mat, 变量 data1)
└── VV_VH_matrices/    # 150 个 VV-VH 交叉相干矩阵 (.mat, 变量 data1)
```
- 类别：PVC、二面角、含水、多分支、电缆、空洞、金属（共 7 类）。
- 每类样本：25 个原始样本 × 6 倍增强 = 150 组。
- 增强方法：旋转(±15°)、缩放(0.8-1.2x)、水平翻转(50%)、增益调整(0.7-1.3x)。
- 每组样本包含：3 种极化图像（HH、VV、VH）、3 种单极化矩阵、3 种交叉相干矩阵。

### 2. 神经网络训练数据集
```
neural_network_dataset/
├── train/
│   ├── VH_images/          # 735 张
│   ├── VV_images/          # 735 张
│   ├── VV_VH_matrices/     # 735 个
│   └── labels.csv
├── test/
│   ├── VH_images/          # 105 张
│   ├── VV_images/          # 105 张
│   ├── VV_VH_matrices/     # 105 个
│   └── labels.csv
├── val/
│   ├── VH_images/          # 210 张
│   ├── VV_images/          # 210 张
│   ├── VV_VH_matrices/     # 210 个
│   └── labels.csv
├── class_mapping.json / class_mapping.txt
└── dataset_report.txt
```
- 总样本数：1,050 组（7 类 × 150 样本/类）。
- 划分比例：训练 70%（735）、测试 10%（105）、验证 20%（210）。
- 类别分布：每个划分内 7 类均衡，样本数完全一致或相差 ≤1。
- **数据范围**：仅包含 VV 图像、VH 图像和 VV_VH 交叉相干矩阵，不包含 HH 相关数据。

## 使用步骤（命令行示例）
```powershell
# 1. 生成训练/验证/测试划分
python -X utf8 reorganize_dataset_for_training.py

# 2. 生成类别映射与统计报告
python -X utf8 generate_dataset_info.py

# 3. 校验 VV/VH 与矩阵的一一对应关系
python -X utf8 verify_correspondence.py

# 4. 测试数据加载器是否正常读取
python -X utf8 test_dataset_loader.py
```
MATLAB 用户可运行同名 `.m` 脚本获得等价结果。

## PyTorch 数据加载示例
```python
from dataset_loader import create_data_loaders

train_loader, val_loader, test_loader, class_names = create_data_loaders(
    dataset_path="neural_network_dataset",
    batch_size=32,
    load_matrices=True
)

for batch in train_loader:
    vh = batch["vh_image"]              # [B, 3, 224, 224]
    vv = batch["vv_image"]              # [B, 3, 224, 224]
    label = batch["class_id"]           # [B]
    coherence = batch["cross_coherence"]# [B, 1, 224, 224]
    # TODO: 模型前向与损失计算
    break
```

## 关键脚本速览
- `calculate_coherence_matrices_224x224.m`：计算 VV-VH 交叉相干矩阵。
- `reorganize_dataset_for_training.py / .m`：按 70/15/15 划分数据集并生成标签。
- `generate_dataset_info.py`：生成 `class_mapping` 与 `dataset_report.txt`。
- `dataset_loader.py`：PyTorch 数据加载器，支持同时返回 VV/VH 图像与交叉相干矩阵。
- `verify_correspondence.py`：验证图像与矩阵的前缀及存在性是否一致。
- `test_dataset_loader.py`：整体连通性测试与统计输出。

## 面向其他 AI 模型的摘要
- **输入模态**：VV 极化图像、VH 极化图像、VV-VH 交叉相干矩阵；分辨率统一为 224×224。
- **样本规模**：总计 1,050 组样本，7 类目标等量分布（每类 150 样本）。训练/测试/验证划分为 735/105/210。
- **增强方法**：采用 6 倍增强组合，包括旋转(±15°)、缩放(0.8-1.2x)、水平翻转(50%)、增益调整(0.7-1.3x)。
- **矩阵含义**：`data1` 为交叉相干幅度 `|S_VV · conj(S_VH)|`，单通道矩阵可直接视为强度图。
- **标签格式**：`labels.csv` 中包含 `sample_id, class_id, class_name, vh_file, vv_file, matrix_file` 字段。
- **目录习惯**：所有文件依照 `000000_class_<id>_TYPE.ext` 命名，便于按编号追踪。
- **推荐用法**：将 VV/VH 图像拼接或分别输入模型，同时将交叉相干矩阵作为另一通道或特征图，用于分类、检测或信号分析任务。

如需将数据交付给其他 AI 模型，只需同步 `neural_network_dataset/` 整体目录及 `class_mapping.json`，即可保证数据结构完整、标签明确、模态清晰。

---

## 🚀 深度学习模型训练（TriBranch Network with CCGA）

### 项目概述
本项目实现了一个基于三分支架构的 SAR 图像分类网络，核心创新点是 **CCGA (Coherence-Guided Attention)** 模块，利用 VV-VH 交叉相干信息引导注意力机制。

### 快速开始

#### 1. 环境准备
```bash
cd network_code
pip install -r requirements.txt
```

#### 2. 训练模型
```bash
# 使用默认配置训练
python train.py --config config.yaml

# 使用改进配置训练
python train.py --config config_improved.yaml
```

#### 3. 评估模型
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 网络架构

**三分支结构**：
- **VV 分支**：ResNet18 提取 VV 极化图像特征
- **VH 分支**：ResNet18 提取 VH 极化图像特征
- **相干分支**：轻量级 CNN 提取 VV-VH 交叉相干特征

**CCGA 模块**（核心创新）：
- 使用交叉相干矩阵作为物理先验引导多头注意力
- 自适应门控机制动态融合学习注意力和物理注意力
- 多尺度相干金字塔增强多尺度特征表达

### 📊 当前实验状态

#### ✅ 已完成的实验

1. **基础消融实验**（6 个实验）
   - 完整模型 (Full)
   - 无 CCGA (No CCGA)
   - 无多尺度 (No Multi-scale)
   - 无选择性一致性损失 (No SC Loss)
   - 固定门控 (Fixed Gate)
   - 乘性融合 (Multiplicative)
   - 报告：`ablation_results/final_results.json`

2. **Baseline vs A2 对比实验**
   - Baseline (行复制亲和矩阵): 100% 验证精度
   - A2 (对称外积亲和矩阵): 99.55% 验证精度
   - 报告：`ablation_results/affinity_comparison_report.md`
   - 结论：Baseline 在当前数据集上表现更好

3. **噪声鲁棒性实验**
   - 测试了 5 种噪声水平（SNR: 30dB, 20dB, 15dB, 10dB, 5dB）
   - 报告：`noise_robustness_report_v2.md`

#### ⏳ 进行中的实验：A1-A4 改进消融实验

**实验目标**：评估 CCGA 模块的 4 项改进措施的有效性

**改进项**：
- **A1**：物理先验作为 Logit Bias（加性偏置而非线性插值）
- **A2**：对称物理亲和矩阵（外积 A_ij = w_i × w_j）
- **A3**：可学习标定（sigmoid 标定代替 min-max 归一化）
- **A4**：增强门控输入（VV+VH+相干统计量）

**实验配置**：8 个实验（Baseline + 4 个单项 + 3 个组合）

**详细指南**：📖 **`ablation_results/A1_A4_ABLATION_STUDY_GUIDE.md`**

**当前状态**：
- ✅ 代码实现完成（`network_code/modules/ccga.py`）
- ✅ 配置文件完成（`network_code/configs_ablation_a1a4/*.yaml`）
- ✅ 自动化脚本完成（`network_code/run_ablation_a1a4.py`）
- ⏳ 等待运行实验（预计 2-4 小时）

**如何继续**：
```bash
cd network_code
python -u run_ablation_a1a4.py
# 输入 'y' 开始实验
```

### 📁 重要文档

| 文档 | 说明 |
|------|------|
| `network_code/README.md` | 网络代码详细说明 |
| `network_code/PROJECT_SUMMARY.md` | 项目完整总结 |
| `network_code/QUICK_GUIDE.md` | 快速上手指南 |
| `ablation_results/A1_A4_ABLATION_STUDY_GUIDE.md` | **A1-A4 消融实验完整指南** ⭐ |
| `ablation_results/affinity_comparison_report.md` | Baseline vs A2 对比报告 |
| `CCGA模块提升识别率方案.md` | CCGA 改进方案（8 项建议） |
| `消融实验完成总结.md` | 基础消融实验总结 |
| `noise_robustness_report_v2.md` | 噪声鲁棒性实验报告 |

### 🎯 给下一个 AI 的快速指引

**如果你需要继续 A1-A4 消融实验**：
1. 阅读 `ablation_results/A1_A4_ABLATION_STUDY_GUIDE.md`
2. 运行 `python -u network_code/run_ablation_a1a4.py`
3. 等待实验完成（2-4 小时）
4. 分析结果并生成最终报告

**如果你需要训练新模型**：
1. 阅读 `network_code/QUICK_GUIDE.md`
2. 修改配置文件 `network_code/config.yaml`
3. 运行 `python network_code/train.py --config network_code/config.yaml`

**如果你需要评估现有模型**：
1. 运行 `python network_code/evaluate.py --checkpoint <模型路径>`
2. 查看 `network_code/eval_results/` 中的结果

### 📊 实验结果概览

**最佳模型性能**（基于 `neural_network_dataset_clean`）：
- 训练精度：~95%
- 验证精度：100%（Baseline 配置，Epoch 13）
- 测试精度：待评估

**数据集统计**：
- 训练集：714 样本
- 验证集：210 样本
- 测试集：105 样本
- 类别数：7 类（PVC、二面角、含水、多分支、电缆、空洞、金属）

### 🔧 技术栈

- **深度学习框架**：PyTorch 1.13+
- **骨干网络**：ResNet18 (预训练)
- **优化器**：AdamW
- **学习率调度**：CosineAnnealingLR
- **数据增强**：旋转、翻转、缩放、Mixup
- **损失函数**：交叉熵 + 选择性一致性损失

---

**最后更新**：2025-10-29
**项目状态**：A1-A4 消融实验配置完成，等待运行
