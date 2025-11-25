"""
轻量级CCGA分类网络
Lightweight CCGA Classification Network

特点：
- 使用轻量级骨干网络（MobileNetV2或EfficientNet-B0）
- 嵌入CCGA模块进行VV和VH特征融合
- 适合快速训练和测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.ccga import CoherenceGuidedAttention


class LightweightCCGANet(nn.Module):
    """轻量级CCGA分类网络"""
    
    def __init__(
        self,
        num_classes=7,
        backbone='mobilenet',  # 'mobilenet' or 'efficientnet' or 'simple_cnn'
        feature_dim=256,
        num_heads=4,
        use_adaptive_gate=True,
        dropout=0.3
    ):
        """
        Args:
            num_classes: 分类类别数
            backbone: 骨干网络类型
            feature_dim: CCGA特征维度
            num_heads: CCGA注意力头数
            use_adaptive_gate: 是否使用自适应门控
            dropout: Dropout比例
        """
        super(LightweightCCGANet, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_type = backbone
        self.feature_dim = feature_dim
        
        # 选择骨干网络
        if backbone == 'mobilenet':
            from torchvision.models import mobilenet_v2
            # VV通道骨干
            self.vv_backbone = mobilenet_v2(pretrained=False)
            self.vv_backbone.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
            self.vv_backbone.classifier = nn.Identity()
            backbone_out_dim = 1280
            
            # VH通道骨干
            self.vh_backbone = mobilenet_v2(pretrained=False)
            self.vh_backbone.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
            self.vh_backbone.classifier = nn.Identity()
            
        elif backbone == 'simple_cnn':
            # 简单CNN骨干（最轻量）
            self.vv_backbone = SimpleCNN()
            self.vh_backbone = SimpleCNN()
            backbone_out_dim = 512
            
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 特征投影到统一维度
        self.vv_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.vh_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # CCGA模块
        self.ccga = CoherenceGuidedAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_adaptive_gate=use_adaptive_gate,
            fusion_mode="gated"
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
    def forward(self, vv_img, vh_img, coherence_matrix):
        """
        前向传播
        Args:
            vv_img: VV图像 (B, 1, H, W)
            vh_img: VH图像 (B, 1, H, W)
            coherence_matrix: 相干矩阵 (B, 1, H, W)
        Returns:
            logits: 分类logits (B, num_classes)
            gate_values: 门控值（用于分析）
        """
        batch_size = vv_img.size(0)

        # 提取VV和VH特征
        vv_feat = self.vv_backbone(vv_img)  # (B, backbone_out_dim)
        vh_feat = self.vh_backbone(vh_img)  # (B, backbone_out_dim)

        # 投影到统一维度
        vv_feat = self.vv_proj(vv_feat)  # (B, feature_dim)
        vh_feat = self.vh_proj(vh_feat)  # (B, feature_dim)

        # 重塑为4D张量用于CCGA (B, C, H, W)
        # 这里我们将特征向量重塑为1x1的特征图
        vv_feat_4d = vv_feat.view(batch_size, self.feature_dim, 1, 1)  # (B, feature_dim, 1, 1)
        vh_feat_4d = vh_feat.view(batch_size, self.feature_dim, 1, 1)  # (B, feature_dim, 1, 1)

        # CCGA融合
        fused_feat_4d = self.ccga(vv_feat_4d, vh_feat_4d, coherence_matrix)

        # 展平回向量
        fused_feat = fused_feat_4d.view(batch_size, self.feature_dim)  # (B, feature_dim)

        # 分类
        logits = self.classifier(fused_feat)

        # 获取gate值（如果CCGA有gate）
        gate_values = None
        if hasattr(self.ccga, 'last_gate_values'):
            gate_values = self.ccga.last_gate_values
        else:
            # 如果没有gate值，返回一个默认值
            gate_values = torch.ones(batch_size, 1).to(vv_img.device) * 0.5

        return logits, gate_values


class SimpleCNN(nn.Module):
    """简单的CNN骨干网络（最轻量）"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(256, 512)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 测试模型
    print('='*80)
    print('轻量级CCGA网络测试')
    print('='*80)

    # 创建模型
    model = LightweightCCGANet(
        num_classes=7,
        backbone='simple_cnn',
        feature_dim=256,
        num_heads=4
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'\n模型参数统计:')
    print(f'  总参数量: {total_params:,}')
    print(f'  可训练参数: {trainable_params:,}')
    print(f'  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)')

    # 测试前向传播
    batch_size = 4
    vv_img = torch.randn(batch_size, 1, 224, 224)
    vh_img = torch.randn(batch_size, 1, 224, 224)
    coherence = torch.randn(batch_size, 1, 224, 224)

    print(f'\n输入形状:')
    print(f'  VV图像: {vv_img.shape}')
    print(f'  VH图像: {vh_img.shape}')
    print(f'  相干矩阵: {coherence.shape}')

    logits, gate_values = model(vv_img, vh_img, coherence)

    print(f'\n输出形状:')
    print(f'  Logits: {logits.shape}')
    print(f'  Gate值: {gate_values.shape}')
    print(f'  Gate值范围: [{gate_values.min():.4f}, {gate_values.max():.4f}]')

    print('\n' + '='*80)
    print('✅ 模型测试通过！')
    print('='*80)

