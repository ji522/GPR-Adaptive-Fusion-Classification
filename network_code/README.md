# 三分支物理引导网络 (Three-Branch Physics-Guided Network)

## 📋 项目概述

这是一个针对GPR（地质雷达）图像分类的深度学习框架，采用创新的三分支物理引导网络架构。

### 核心创新点

1. **三分支物理引导网络架构**
   - VV极化分支
   - VH极化分支
   - 交叉相干分支（物理引导）

2. **相干引导注意力机制 (CCGA)** - 可选模块
   - 使用物理相干矩阵直接生成注意力权重
   - 高相干区域获得高权重，低相干区域获得低权重

3. **选择性一致性损失函数** - 可选模块
   - 仅在高相干区域强制VV-VH特征对齐
   - 避免在低相干（噪声）区域施加不合理约束

4. **多尺度相干金字塔** - 可选模块
   - 在多个尺度应用物理相干性指导
   - 实现全方位多粒度特征优化

## 📁 项目结构

```
network_code/
├── __init__.py                          # 包初始化
├── config.yaml                          # 配置文件
├── config.py                            # 配置加载器
├── utils.py                             # 工具函数
├── train.py                             # 训练脚本
├── evaluate.py                          # 评估脚本
├── inference.py                         # 推理脚本
├── models/
│   ├── __init__.py
│   ├── backbones.py                     # Backbone网络和相干分支
│   └── tribranch_network.py             # 三分支网络主模型
├── losses/
│   ├── __init__.py
│   ├── classification_loss.py           # 分类损失
│   ├── selective_consistency_loss.py    # 选择性一致性损失
│   └── combined_loss.py                 # 组合损失
├── modules/
│   ├── __init__.py
│   ├── ccga.py                          # 相干引导注意力机制
│   └── multi_scale_pyramid.py           # 多尺度相干金字塔
├── logs/                                # TensorBoard日志
├── checkpoints/                         # 模型检查点
└── README.md                            # 本文件
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install torch torchvision pytorch-lightning
pip install pyyaml scikit-learn matplotlib seaborn
pip install scipy pillow pandas numpy
```

### 2. 配置文件

编辑 `config.yaml` 配置训练参数：

```yaml
# 数据集配置
dataset:
  root: "neural_network_dataset"
  batch_size: 32
  num_workers: 4

# 模型配置
model:
  backbone: "resnet50"
  pretrained: true

# 训练配置
training:
  epochs: 100
  optimizer:
    type: "AdamW"
    lr: 0.001
```

### 3. 训练模型

#### 基础三分支网络（第一阶段）

```bash
# 使用默认配置训练
python network_code/train.py

# 使用自定义配置
python network_code/train.py --config network_code/config.yaml --seed 42

# 恢复训练
python network_code/train.py --resume network_code/checkpoints/best_model.pth
```

#### 启用高级模块（第二阶段）

编辑 `config.yaml` 启用可选模块：

```yaml
# 启用相干引导注意力机制
model:
  ccga:
    enabled: true
    num_heads: 8

# 启用选择性一致性损失
loss:
  selective_consistency:
    enabled: true
    weight: 0.5
    coherence_threshold: 0.7

# 启用多尺度相干金字塔
model:
  multi_scale:
    enabled: true
    scales: [1, 2, 4]
```

### 4. 评估模型

```bash
# 评估测试集
python network_code/evaluate.py \
  --config network_code/config.yaml \
  --checkpoint network_code/checkpoints/best_model.pth \
  --save-dir network_code/eval_results
```

### 5. 推理

```bash
# 单个样本推理
python network_code/inference.py \
  --config network_code/config.yaml \
  --checkpoint network_code/checkpoints/best_model.pth \
  --vv-image path/to/vv_image.jpg \
  --vh-image path/to/vh_image.jpg \
  --coherence-matrix path/to/coherence_matrix.mat
```

## 📊 数据格式

### 输入数据

- **VV图像**: 224×224 RGB JPEG
- **VH图像**: 224×224 RGB JPEG
- **相干矩阵**: 224×224 MATLAB .mat 文件（变量名: `data1`）

### 数据集结构

```
neural_network_dataset/
├── train/
│   ├── VV_images/          # 735张VV图像
│   ├── VH_images/          # 735张VH图像
│   ├── VV_VH_matrices/     # 735个相干矩阵
│   └── labels.csv
├── val/
│   ├── VV_images/          # 154张
│   ├── VH_images/          # 154张
│   ├── VV_VH_matrices/     # 154个
│   └── labels.csv
├── test/
│   ├── VV_images/          # 161张
│   ├── VH_images/          # 161张
│   ├── VV_VH_matrices/     # 161个
│   └── labels.csv
└── class_mapping.json
```

## 🔧 模型架构

### 三分支网络

```
输入: VV图像 + VH图像 + 相干矩阵
     ↓          ↓           ↓
  VV分支    VH分支    相干分支
  (ResNet)  (ResNet)  (轻量级CNN)
     ↓          ↓           ↓
  特征提取  特征提取   特征提取
     └──────┬──────┘
            ↓
      特征拼接与融合
            ↓
         分类器
            ↓
         输出 (7类)
```

### 可选模块

#### 相干引导注意力机制 (CCGA)
- 多头注意力
- 物理相干性引导
- 可学习温度参数

#### 选择性一致性损失
- 基于相干阈值的区域选择
- 支持多种距离度量 (cosine, L2, L1)
- 自适应阈值学习

#### 多尺度相干金字塔
- 多尺度特征融合
- 支持多种融合方法 (weighted_sum, concat, attention)
- 全方位多粒度特征优化

## 📈 训练监控

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir network_code/logs

# 访问 http://localhost:6006
```

### 日志输出

训练过程中会输出：
- 每个batch的损失和准确率
- 每个epoch的验证指标
- 学习率变化
- 最佳模型信息

## 🎯 性能指标

评估脚本会生成：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵
- 详细分类报告

## 💡 使用建议

### 第一阶段：基础训练
1. 使用默认配置训练基础三分支网络
2. 验证数据加载和模型前向传播
3. 观察基础性能

### 第二阶段：模块集成
1. 逐步启用可选模块
2. 调整模块参数
3. 对比性能提升

### 第三阶段：超参数优化
1. 调整学习率、批次大小等
2. 尝试不同的backbone
3. 优化损失函数权重

## 🔍 故障排除

### 显存不足
- 减小 `batch_size`
- 使用更小的 backbone (resnet18/34)
- 启用混合精度训练

### 训练不收敛
- 检查学习率设置
- 验证数据加载是否正确
- 尝试不同的优化器

### 模型过拟合
- 增加 dropout 比例
- 使用数据增强
- 启用早停策略

## 📚 参考文献

- ResNet: He et al., "Deep Residual Learning for Image Recognition"
- Attention: Vaswani et al., "Attention Is All You Need"
- 物理引导深度学习: 相关研究论文

## 📝 许可证

MIT License

## 👥 贡献

欢迎提交问题和改进建议！

---

**最后更新**: 2025-10-20

