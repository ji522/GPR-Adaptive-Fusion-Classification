"""
轻量级CCGA模型训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 添加路径
sys.path.append('network_code')
from models.lightweight_ccga_net import LightweightCCGANet
from dataset_loader import create_dataloaders

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_gate_evolution(gate_history, save_dir):
    """
    绘制门控值演化曲线
    展示深度学习比例和物理先验比例随训练的变化
    """
    epochs = gate_history['epoch']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 门控值变化曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, gate_history['train_gate_mean'], 'b-o', label='Train Gate', linewidth=2, markersize=4)
    ax1.plot(epochs, gate_history['test_gate_mean'], 'r-s', label='Test Gate', linewidth=2, markersize=4)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Gate Value', fontsize=12)
    ax1.set_title('Gate Value Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 深度学习 vs 物理先验比例（训练集）
    ax2 = axes[0, 1]
    ax2.plot(epochs, [r*100 for r in gate_history['train_dl_ratio']],
             'b-o', label='Deep Learning', linewidth=2, markersize=4)
    ax2.plot(epochs, [r*100 for r in gate_history['train_physics_ratio']],
             'g-^', label='Physics Prior', linewidth=2, markersize=4)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Ratio (%)', fontsize=12)
    ax2.set_title('Train Set: DL vs Physics Ratio', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    # 3. 深度学习 vs 物理先验比例（测试集）
    ax3 = axes[1, 0]
    ax3.plot(epochs, [r*100 for r in gate_history['test_dl_ratio']],
             'b-o', label='Deep Learning', linewidth=2, markersize=4)
    ax3.plot(epochs, [r*100 for r in gate_history['test_physics_ratio']],
             'g-^', label='Physics Prior', linewidth=2, markersize=4)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Ratio (%)', fontsize=12)
    ax3.set_title('Test Set: DL vs Physics Ratio', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])

    # 4. 堆叠面积图（测试集）
    ax4 = axes[1, 1]
    ax4.fill_between(epochs, 0, [r*100 for r in gate_history['test_dl_ratio']],
                     alpha=0.6, color='blue', label='Deep Learning')
    ax4.fill_between(epochs, [r*100 for r in gate_history['test_dl_ratio']], 100,
                     alpha=0.6, color='green', label='Physics Prior')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Ratio (%)', fontsize=12)
    ax4.set_title('Test Set: Contribution Ratio (Stacked)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_ylim([0, 100])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gate_evolution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'  门控值演化图已保存: {save_path}')

    # 保存数值数据
    import json
    data_path = os.path.join(save_dir, 'gate_history.json')
    # 转换numpy类型为Python原生类型
    gate_history_serializable = {}
    for key, value in gate_history.items():
        if isinstance(value, list):
            gate_history_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        else:
            gate_history_serializable[key] = value
    with open(data_path, 'w') as f:
        json.dump(gate_history_serializable, f, indent=2)
    print(f'  门控值数据已保存: {data_path}')


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    all_gates = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for vv_img, vh_img, coherence, labels in pbar:
        # 移动到设备
        vv_img = vv_img.to(device)
        vh_img = vh_img.to(device)
        coherence = coherence.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        logits, gate_values = model(vv_img, vh_img, coherence)

        # 分类损失
        cls_loss = criterion(logits, labels)

        # Gate多样性损失：鼓励不同类别使用不同的gate值
        # 计算每个类别的平均gate值，然后最大化类别间的方差
        gate_diversity_loss = 0.0
        if len(torch.unique(labels)) > 1:  # 至少有2个类别
            class_gates = []
            for cls in torch.unique(labels):
                mask = labels == cls
                if mask.sum() > 0:
                    class_gate_mean = gate_values[mask].mean()
                    class_gates.append(class_gate_mean)

            if len(class_gates) > 1:
                class_gates_tensor = torch.stack(class_gates)
                # 负方差作为损失（最大化方差 = 最小化负方差）
                gate_diversity_loss = -torch.var(class_gates_tensor) * 0.02  # 适中的权重

        # Gate饱和惩罚：防止gate值完全饱和到0或1
        # 当gate接近0或1时，给予轻微惩罚，但允许接近（比如0.1~0.9）
        # 使用平滑的惩罚函数：只惩罚非常极端的值（<0.05或>0.95）
        gate_saturation_penalty = (
            torch.relu(0.05 - gate_values).mean() +  # 惩罚<0.05的值
            torch.relu(gate_values - 0.95).mean()     # 惩罚>0.95的值
        ) * 0.5  # 轻微惩罚

        # 总损失
        loss = cls_loss + gate_diversity_loss + gate_saturation_penalty

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_gates.extend(gate_values.cpu().detach().numpy().flatten())

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'gate': f'{gate_values.mean().item():.3f}'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_gate = np.mean(all_gates)

    return avg_loss, accuracy, avg_gate, all_gates


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_gates = []
    
    with torch.no_grad():
        for vv_img, vh_img, coherence, labels in tqdm(test_loader, desc='Evaluating'):
            vv_img = vv_img.to(device)
            vh_img = vh_img.to(device)
            coherence = coherence.to(device)
            labels = labels.to(device)
            
            logits, gate_values = model(vv_img, vh_img, coherence)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gates.extend(gate_values.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels, all_gates


def train(args):
    """主训练函数"""
    print('='*80)
    print('轻量级CCGA模型训练')
    print('='*80)
    print(f'数据集: {args.dataset}')
    print(f'骨干网络: {args.backbone}')
    print(f'批次大小: {args.batch_size}')
    print(f'学习率: {args.lr}')
    print(f'训练轮数: {args.epochs}')
    print('='*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 创建数据加载器
    print('\n加载数据集...')
    train_loader, test_loader = create_dataloaders(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    
    print(f'训练集: {len(train_loader.dataset)} 样本')
    print(f'测试集: {len(test_loader.dataset)} 样本')
    
    # 创建模型
    print('\n创建模型...')
    model = LightweightCCGANet(
        num_classes=7,
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        use_adaptive_gate=True,
        dropout=args.dropout
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,} ({total_params*4/1024/1024:.2f} MB)')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    log_dir = os.path.join(args.save_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)

    # 训练循环
    print('\n开始训练...\n')
    best_acc = 0

    # 用于记录gate值历史
    gate_history = {
        'epoch': [],
        'train_gate_mean': [],
        'test_gate_mean': [],
        'train_dl_ratio': [],  # 深度学习比例
        'train_physics_ratio': [],  # 物理先验比例
        'test_dl_ratio': [],
        'test_physics_ratio': []
    }

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc, train_gate_mean, train_gates = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 评估
        test_loss, test_acc, preds, labels, test_gates = evaluate(model, test_loader, criterion, device)

        # 学习率调整
        scheduler.step()

        # 计算深度学习和物理先验的比例
        train_dl_ratio = train_gate_mean  # gate值就是深度学习的比例
        train_physics_ratio = 1 - train_gate_mean  # 1-gate就是物理先验的比例
        test_dl_ratio = np.mean(test_gates)
        test_physics_ratio = 1 - test_dl_ratio

        # 记录历史
        gate_history['epoch'].append(epoch)
        gate_history['train_gate_mean'].append(train_gate_mean)
        gate_history['test_gate_mean'].append(np.mean(test_gates))
        gate_history['train_dl_ratio'].append(train_dl_ratio)
        gate_history['train_physics_ratio'].append(train_physics_ratio)
        gate_history['test_dl_ratio'].append(test_dl_ratio)
        gate_history['test_physics_ratio'].append(test_physics_ratio)

        # TensorBoard记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Gate/train_mean', train_gate_mean, epoch)
        writer.add_scalar('Gate/test_mean', np.mean(test_gates), epoch)
        writer.add_scalar('Ratio/train_deep_learning', train_dl_ratio, epoch)
        writer.add_scalar('Ratio/train_physics_prior', train_physics_ratio, epoch)
        writer.add_scalar('Ratio/test_deep_learning', test_dl_ratio, epoch)
        writer.add_scalar('Ratio/test_physics_prior', test_physics_ratio, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 打印结果
        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'  Gate Mean: {np.mean(test_gates):.3f} ± {np.std(test_gates):.3f}')
        print(f'  深度学习比例: {test_dl_ratio:.1%}, 物理先验比例: {test_physics_ratio:.1%}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, save_path)
            print(f'  ✅ 保存最佳模型 (Acc: {test_acc:.2f}%)')

    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
    }, final_path)

    writer.close()

    # 绘制门控值变化曲线
    print('\n生成门控值可视化...')
    plot_gate_evolution(gate_history, args.save_dir)

    print('\n' + '='*80)
    print('训练完成！')
    print('='*80)
    print(f'最佳测试准确率: {best_acc:.2f}%')
    print(f'模型保存路径: {args.save_dir}')
    print(f'门控值可视化: {os.path.join(args.save_dir, "gate_evolution.png")}')
    print('='*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='轻量级CCGA模型训练')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='tongyi_weidu_10x', help='数据集目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    # 模型参数
    parser.add_argument('--backbone', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'mobilenet'], help='骨干网络')
    parser.add_argument('--feature_dim', type=int, default=256, help='特征维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='experiments/lightweight_ccga', help='保存目录')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 开始训练
    train(args)

