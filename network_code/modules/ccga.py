"""
相干引导注意力机制 (CCGA) - 样本自适应门控版本
Coherence-Guided Attention (CCGA) with Adaptive Gating

创新点：
- 使用物理相干矩阵引导注意力权重
- 样本自适应门控融合：网络学习何时依赖物理先验
- 高相干区域获得高权重，低相干区域获得低权重
- 物理可解释的注意力分配

改进历史：
- v1.0: 乘性融合 (scores * coherence_weights)
- v2.0: 样本自适应门控融合 (gate * scores_learned + (1-gate) * scores_physical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoherenceGuidedAttention(nn.Module):
    """
    相干引导注意力机制（样本自适应门控版本）

    基于交叉相干矩阵生成注意力权重，指导VV和VH特征的融合
    使用门控机制自适应平衡数据驱动学习和物理先验
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temperature: bool = True,
        use_adaptive_gate: bool = True,
        gate_hidden_dim: int = 64,
        fusion_mode: str = "gated",  # "gated" or "multiplicative"
        physical_affinity: str = "repeat",  # "repeat" or "symmetric_outer"
        # A1: 物理作为 logit bias
        use_logit_bias: bool = False,
        logit_bias_lambda: float = 1.0,
        # A3: 可学习标定
        use_calibration: bool = False,
        calibration_init_a: float = 5.0,
        calibration_init_b: float = 0.5,
        # A4: 增强门控输入
        use_enhanced_gate: bool = False,
    ):
        """
        Args:
            feature_dim: 特征维度
            num_heads: 多头注意力的头数
            dropout: Dropout比例
            use_temperature: 是否使用温度参数
            use_adaptive_gate: 是否使用样本自适应门控（推荐True）
            gate_hidden_dim: 门控预测网络的隐藏层维度
            fusion_mode: 融合模式 ("gated" for v2.0, "multiplicative" for v1.0)
            physical_affinity: 物理亲和矩阵的构造方式 ("repeat" 或 "symmetric_outer")
            use_logit_bias: A1改进 - 物理先验作为logit偏置而非线性插值
            logit_bias_lambda: A1改进 - logit偏置的权重系数
            use_calibration: A3改进 - 使用可学习sigmoid标定代替min-max归一化
            calibration_init_a: A3改进 - sigmoid标定的初始斜率
            calibration_init_b: A3改进 - sigmoid标定的初始偏移
            use_enhanced_gate: A4改进 - 门控输入包含VV+VH+相干统计
        """
        super(CoherenceGuidedAttention, self).__init__()

        assert feature_dim % num_heads == 0, "feature_dim必须能被num_heads整除"
        assert fusion_mode in ["gated", "multiplicative"], "fusion_mode必须是'gated'或'multiplicative'"
        assert physical_affinity in ["repeat", "symmetric_outer"], "physical_affinity必须是'repeat'或'symmetric_outer'"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_adaptive_gate = use_adaptive_gate
        self.fusion_mode = fusion_mode
        self.physical_affinity = physical_affinity

        # A1: logit bias 配置
        self.use_logit_bias = use_logit_bias
        self.logit_bias_lambda = logit_bias_lambda

        # A3: 可学习标定配置
        self.use_calibration = use_calibration

        # A4: 增强门控配置
        self.use_enhanced_gate = use_enhanced_gate

        # 线性投影
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        # 输出投影
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 温度参数（可学习）
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1) * (self.head_dim ** -0.5))
        else:
            self.register_buffer('temperature', torch.ones(1) * (self.head_dim ** -0.5))

        # A3: 可学习标定参数
        if use_calibration:
            self.calibration_a = nn.Parameter(torch.tensor(calibration_init_a))
            self.calibration_b = nn.Parameter(torch.tensor(calibration_init_b))
            print(f"✓ CCGA-A3: 启用可学习标定 (init_a={calibration_init_a}, init_b={calibration_init_b})")

        # 门控机制（仅在gated模式下使用）
        if fusion_mode == "gated":
            if use_adaptive_gate:
                # A4: 增强门控输入
                if use_enhanced_gate:
                    # 门控输入：VV pooled + VH pooled + 相干统计 (mean, std, max)
                    gate_input_dim = feature_dim * 2 + 3  # VV + VH + 3个统计量
                    self.gate_predictor = nn.Sequential(
                        nn.Linear(gate_input_dim, gate_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(gate_hidden_dim, 1),
                        nn.Sigmoid()
                    )
                    print(f"✓ CCGA-A4: 启用增强门控输入 (VV+VH+相干统计, hidden_dim={gate_hidden_dim})")
                else:
                    # 样本自适应门控：根据VV和VH特征以及相干矩阵预测gate值
                    # 输入：VV特征 + VH特征 + 相干矩阵统计 = feature_dim * 2 + 3
                    gate_input_dim = feature_dim * 2 + 3
                    self.gate_predictor = nn.Sequential(
                        nn.Linear(gate_input_dim, gate_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(gate_hidden_dim, gate_hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                        nn.Linear(gate_hidden_dim // 2, 1),
                        nn.Sigmoid()               # [B, 1] 输出范围[0, 1]
                    )
                    print(f"✓ CCGA: 启用样本自适应门控 (输入=VV+VH+相干统计, gate_hidden_dim={gate_hidden_dim})")
            else:
                # 全局门控：所有样本共享一个gate值
                self.gate_weight = nn.Parameter(torch.tensor(0.5))
                print(f"✓ CCGA: 启用全局门控 (初始值=0.5)")
        else:
            print(f"✓ CCGA: 使用乘性融合 (v1.0)")
    
    def forward(
        self,
        vv_features: torch.Tensor,
        vh_features: torch.Tensor,
        coherence_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播（样本自适应门控版本）

        Args:
            vv_features: VV特征 [B, C, H, W]
            vh_features: VH特征 [B, C, H, W]
            coherence_matrix: 交叉相干矩阵 [B, 1, H_orig, W_orig]

        Returns:
            融合特征 [B, C, H, W]
        """
        B, C, H, W = vv_features.shape

        # 将特征展平为序列形式
        vv_seq = vv_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        vh_seq = vh_features.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 调整相干矩阵到特征图尺寸
        coherence_resized = F.interpolate(
            coherence_matrix,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # [B, 1, H, W]

        # 归一化相干矩阵到[0, 1]
        coherence_flat = coherence_resized.flatten(2).transpose(1, 2)  # [B, H*W, 1]

        # A3: 可学习标定 vs 传统 min-max 归一化
        if self.use_calibration:
            # 使用可学习的 sigmoid 标定
            coherence_clamped = coherence_flat.clamp(0, 1)
            coherence_norm = torch.sigmoid(
                self.calibration_a * (coherence_clamped - self.calibration_b)
            ).clamp(1e-4, 1 - 1e-4)  # [B, H*W, 1]
        else:
            # 传统 min-max 归一化
            coherence_min = coherence_flat.min(dim=1, keepdim=True)[0]
            coherence_max = coherence_flat.max(dim=1, keepdim=True)[0]
            coherence_norm = (coherence_flat - coherence_min) / (coherence_max - coherence_min + 1e-8)

        # 生成相干引导的注意力权重（物理先验）
        coherence_weights = coherence_norm  # [B, H*W, 1]

        # 计算查询、键、值
        Q = self.query(vv_seq)  # [B, H*W, C]
        K = self.key(vh_seq)    # [B, H*W, C]
        V = self.value(vh_seq)  # [B, H*W, C]

        # 重塑为多头形式
        Q = Q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        K = K.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        V = V.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]

        # 计算数据驱动的注意力分数
        scores_learned = torch.matmul(Q, K.transpose(-2, -1)) * self.temperature  # [B, num_heads, H*W, H*W]

        # 计算物理引导的注意力分数
        N = H * W
        if self.physical_affinity == "symmetric_outer":
            # 对称外积亲和矩阵 A_ij = w_i * w_j
            w = coherence_weights.squeeze(-1)  # [B, N]
            A = torch.einsum('bi,bj->bij', w, w)  # [B, N, N]
            scores_physical = A.unsqueeze(1).expand(B, self.num_heads, N, N)
        else:
            # 行复制：每行相同，仅依赖被关注位置 j 的可靠度
            coherence_weights_expanded = coherence_weights.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 1]
            coherence_weights_expanded = coherence_weights_expanded.expand(B, self.num_heads, N, N, 1)
            scores_physical = coherence_weights_expanded.squeeze(-1)  # [B, num_heads, N, N]

        # 融合策略
        if self.fusion_mode == "gated":
            # v2.0: 门控融合 - 自适应平衡数据学习和物理先验
            if self.use_adaptive_gate:
                # A4: 增强门控输入
                if self.use_enhanced_gate:
                    # 计算 VV 和 VH 的全局池化特征
                    vv_pooled = F.adaptive_avg_pool2d(vv_features, 1).flatten(1)  # [B, C]
                    vh_pooled = F.adaptive_avg_pool2d(vh_features, 1).flatten(1)  # [B, C]

                    # 计算相干矩阵的全局统计量
                    # coherence_weights: [B, H*W, 1] -> squeeze to [B, H*W] for statistics
                    coh_squeezed = coherence_weights.squeeze(-1)  # [B, H*W]
                    coh_mean = coh_squeezed.mean(dim=1, keepdim=True)  # [B, 1]
                    coh_std = coh_squeezed.std(dim=1, keepdim=True)    # [B, 1]
                    coh_max = coh_squeezed.max(dim=1, keepdim=True)[0]  # [B, 1]

                    # 拼接所有输入
                    gate_input = torch.cat([vv_pooled, vh_pooled, coh_mean, coh_std, coh_max], dim=1)
                    gate = self.gate_predictor(gate_input)  # [B, 1]
                    gate = gate.view(B, 1, 1, 1)
                else:
                    # 样本自适应门控：根据VV、VH特征和相干矩阵统计预测gate值
                    # 计算 VV 和 VH 的全局池化特征
                    vv_pooled = F.adaptive_avg_pool2d(vv_features, 1).flatten(1)  # [B, C]
                    vh_pooled = F.adaptive_avg_pool2d(vh_features, 1).flatten(1)  # [B, C]

                    # 计算相干矩阵的全局统计量
                    coh_squeezed = coherence_weights.squeeze(-1)  # [B, H*W]
                    coh_mean = coh_squeezed.mean(dim=1, keepdim=True)  # [B, 1]
                    # 使用unbiased=False避免单元素时的NaN
                    coh_std = coh_squeezed.std(dim=1, keepdim=True, unbiased=False)    # [B, 1]
                    coh_max = coh_squeezed.max(dim=1, keepdim=True)[0]  # [B, 1]

                    # 拼接所有输入：VV + VH + 相干统计
                    gate_input = torch.cat([vv_pooled, vh_pooled, coh_mean, coh_std, coh_max], dim=1)
                    gate = self.gate_predictor(gate_input)  # [B, 1]
                    gate = gate.view(B, 1, 1, 1)  # 广播到 [B, num_heads, H*W, H*W]
            else:
                # 全局门控：所有样本共享一个gate值
                gate = torch.sigmoid(self.gate_weight)

            # A1: logit bias vs 线性插值
            if self.use_logit_bias:
                # 物理先验作为 logit 偏置加到学习分数上
                scores = (scores_learned / self.temperature) + self.logit_bias_lambda * scores_physical
            else:
                # 门控融合公式（线性插值）：
                # gate=1: 完全依赖数据学习
                # gate=0: 完全依赖物理先验
                # gate=0.5: 平衡两者
                scores = gate * scores_learned + (1 - gate) * scores_physical
        else:
            # v1.0: 乘性融合 - 直接将物理先验乘到学习的注意力分数上
            scores = scores_learned * scores_physical

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, H*W, H*W]
        attn_weights = self.dropout(attn_weights)

        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, H*W, head_dim]

        # 重塑回原始形式
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, H*W, num_heads, head_dim]
        attn_output = attn_output.view(B, H*W, C)  # [B, H*W, C]

        # 输出投影
        output = self.out_proj(attn_output)  # [B, H*W, C]

        # 重塑回特征图形式
        output = output.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        # 融合VV特征和注意力输出
        fused = vv_features + output

        # 保存gate值用于分析
        if self.fusion_mode == "gated" and self.use_adaptive_gate:
            self.last_gate_values = gate.view(B, -1)  # [B, 1]
        else:
            self.last_gate_values = None

        return fused


class MultiHeadCoherenceAttention(nn.Module):
    """
    多头相干引导注意力
    支持多个注意力头并行处理
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: 特征维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super(MultiHeadCoherenceAttention, self).__init__()
        
        self.attention_heads = nn.ModuleList([
            CoherenceGuidedAttention(
                feature_dim=feature_dim,
                num_heads=1,
                dropout=dropout,
                use_temperature=True,
            )
            for _ in range(num_heads)
        ])
        
        self.fusion = nn.Linear(feature_dim * num_heads, feature_dim)
    
    def forward(
        self,
        vv_features: torch.Tensor,
        vh_features: torch.Tensor,
        coherence_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vv_features: VV特征 [B, C, H, W]
            vh_features: VH特征 [B, C, H, W]
            coherence_matrix: 交叉相干矩阵 [B, 1, H_orig, W_orig]
            
        Returns:
            融合特征 [B, C, H, W]
        """
        B, C, H, W = vv_features.shape
        
        # 多头并行处理
        head_outputs = []
        for head in self.attention_heads:
            head_output = head(vv_features, vh_features, coherence_matrix)
            head_outputs.append(head_output)
        
        # 拼接多头输出
        multi_head_output = torch.cat(head_outputs, dim=1)  # [B, C*num_heads, H, W]
        
        # 融合
        multi_head_output = multi_head_output.flatten(2).transpose(1, 2)  # [B, H*W, C*num_heads]
        fused = self.fusion(multi_head_output)  # [B, H*W, C]
        fused = fused.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        
        return fused


if __name__ == "__main__":
    # 测试相干引导注意力
    print("测试相干引导注意力机制...")
    
    batch_size = 2
    feature_dim = 512
    H, W = 7, 7
    
    # 创建模拟数据
    vv_features = torch.randn(batch_size, feature_dim, H, W)
    vh_features = torch.randn(batch_size, feature_dim, H, W)
    coherence_matrix = torch.rand(batch_size, 1, 224, 224)
    
    # 测试单头注意力
    print("\n1. 单头相干引导注意力...")
    ccga = CoherenceGuidedAttention(
        feature_dim=feature_dim,
        num_heads=8,
        dropout=0.1,
        use_temperature=True,
    )
    output = ccga(vv_features, vh_features, coherence_matrix)
    print(f"✓ 输出形状: {output.shape}")
    print(f"  预期形状: {vv_features.shape}")
    
    # 测试多头注意力
    print("\n2. 多头相干引导注意力...")
    mhca = MultiHeadCoherenceAttention(
        feature_dim=feature_dim,
        num_heads=4,
        dropout=0.1,
    )
    output_mh = mhca(vv_features, vh_features, coherence_matrix)
    print(f"✓ 输出形状: {output_mh.shape}")
    print(f"  预期形状: {vv_features.shape}")
    
    # 测试梯度反向传播
    print("\n3. 梯度反向传播...")
    loss = output.sum() + output_mh.sum()
    loss.backward()
    print(f"✓ 梯度反向传播成功")
    
    print("\n✓ 所有测试通过！")

