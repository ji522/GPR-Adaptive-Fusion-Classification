"""
轻量级CCGA模型测试脚本
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, confusion_matrix
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


def test(args):
    """测试模型"""
    print('='*80)
    print('轻量级CCGA模型测试')
    print('='*80)
    print(f'数据集: {args.dataset}')
    print(f'模型路径: {args.model_path}')
    print('='*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 创建数据加载器
    print('\n加载数据集...')
    _, test_loader = create_dataloaders(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        random_seed=42
    )
    
    print(f'测试集: {len(test_loader.dataset)} 样本')
    
    # 创建模型
    print('\n创建模型...')
    model = LightweightCCGANet(
        num_classes=7,
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        use_adaptive_gate=True,
        dropout=0.0  # 测试时不使用dropout
    ).to(device)
    
    # 加载模型权重
    print(f'\n加载模型权重: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'模型训练轮数: {checkpoint["epoch"]}')
    print(f'模型测试准确率: {checkpoint["test_acc"]:.2f}%')
    
    # 测试
    print('\n开始测试...\n')
    model.eval()
    
    all_preds = []
    all_labels = []
    all_gates = []
    class_gates = {i: [] for i in range(7)}
    
    with torch.no_grad():
        for vv_img, vh_img, coherence, labels in tqdm(test_loader, desc='Testing'):
            vv_img = vv_img.to(device)
            vh_img = vh_img.to(device)
            coherence = coherence.to(device)
            labels = labels.to(device)
            
            logits, gate_values = model(vv_img, vh_img, coherence)
            _, predicted = logits.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_gates.extend(gate_values.cpu().numpy())
            
            # 按类别统计gate值
            for i in range(len(labels)):
                class_gates[labels[i].item()].append(gate_values[i].item())
    
    # 计算准确率
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    print('\n' + '='*80)
    print('测试结果')
    print('='*80)
    print(f'总体准确率: {accuracy:.2f}%')
    print(f'Gate均值: {np.mean(all_gates):.3f} ± {np.std(all_gates):.3f}')
    
    # 类别名称
    class_names = ['PVC', '二面角', '含水', '多分支', '电缆', '空洞', '金属']
    
    # 分类报告
    print('\n分类报告:')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    # 每个类别的Gate值统计
    print('\n每个类别的Gate值统计:')
    print(f'{"类别":<10} {"Gate均值":<12} {"Gate标准差":<12} {"Gate范围":<20} {"深度学习%":<12} {"物理先验%":<12}')
    print('-'*80)
    for i, cls_name in enumerate(class_names):
        if len(class_gates[i]) > 0:
            mean_gate = np.mean(class_gates[i])
            std_gate = np.std(class_gates[i])
            min_gate = np.min(class_gates[i])
            max_gate = np.max(class_gates[i])
            gate_range = f'{min_gate:.4f}~{max_gate:.4f}'
            dl_ratio = mean_gate * 100
            physics_ratio = (1 - mean_gate) * 100
            print(f'{cls_name:<10} {mean_gate:<12.3f} {std_gate:<12.6f} {gate_range:<20} {dl_ratio:<12.1f} {physics_ratio:<12.1f}')

    # 打印所有样本的gate值范围
    all_gates_array = np.array(all_gates)
    print(f'\n所有样本的Gate值统计:')
    print(f'  最小值: {np.min(all_gates_array):.6f}')
    print(f'  最大值: {np.max(all_gates_array):.6f}')
    print(f'  范围: {np.max(all_gates_array) - np.min(all_gates_array):.6f}')
    print(f'  标准差: {np.std(all_gates_array):.6f}')

    # 打印每个样本的gate值（前20个）
    print(f'\n前20个样本的Gate值:')
    for i in range(min(20, len(all_gates))):
        gate_val = all_gates[i]
        if isinstance(gate_val, np.ndarray):
            gate_val = gate_val.item()
        print(f'  样本{i+1}: {gate_val:.6f}')
    
    # 保存结果
    if args.save_results:
        save_dir = os.path.dirname(args.model_path)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
        print(f'\n混淆矩阵已保存: {os.path.join(save_dir, "confusion_matrix.png")}')
        
        # Gate值分布和深度学习/物理先验比例
        fig = plt.figure(figsize=(18, 10))

        # 1. Gate值分布直方图
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(all_gates, bins=50, edgecolor='black', alpha=0.7, facecolor='steelblue')
        plt.axvline(x=np.mean(all_gates), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(all_gates):.3f}')
        plt.axvline(x=0.5, color='gray', linestyle='--',
                   linewidth=2, alpha=0.5, label='Balanced (0.5)')
        plt.xlabel('Gate Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Gate Value Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 2. 每个类别的Gate值箱线图
        ax2 = plt.subplot(2, 3, 2)
        gate_data = [class_gates[i] for i in range(7)]
        bp = plt.boxplot(gate_data, labels=class_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.ylabel('Gate Value', fontsize=12)
        plt.title('Gate Value by Class', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # 3. 深度学习 vs 物理先验比例（饼图）
        ax3 = plt.subplot(2, 3, 3)
        avg_gate = np.mean(all_gates)
        dl_ratio = avg_gate * 100
        physics_ratio = (1 - avg_gate) * 100
        colors = ['#3498db', '#2ecc71']
        explode = (0.05, 0.05)
        plt.pie([dl_ratio, physics_ratio],
               labels=['Deep Learning', 'Physics Prior'],
               autopct='%1.1f%%',
               colors=colors,
               explode=explode,
               shadow=True,
               startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
        plt.title('Overall DL vs Physics Ratio', fontsize=14, fontweight='bold')

        # 4. 每个类别的深度学习比例柱状图
        ax4 = plt.subplot(2, 3, 4)
        class_dl_ratios = [np.mean(class_gates[i]) * 100 for i in range(7)]
        class_physics_ratios = [(1 - np.mean(class_gates[i])) * 100 for i in range(7)]
        x = np.arange(len(class_names))
        width = 0.35
        plt.bar(x - width/2, class_dl_ratios, width, label='Deep Learning',
               color='#3498db', alpha=0.8)
        plt.bar(x + width/2, class_physics_ratios, width, label='Physics Prior',
               color='#2ecc71', alpha=0.8)
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Ratio (%)', fontsize=12)
        plt.title('DL vs Physics Ratio by Class', fontsize=14, fontweight='bold')
        plt.xticks(x, class_names, rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim([0, 100])

        # 5. 堆叠柱状图
        ax5 = plt.subplot(2, 3, 5)
        plt.bar(class_names, class_dl_ratios, label='Deep Learning',
               color='#3498db', alpha=0.8)
        plt.bar(class_names, class_physics_ratios, bottom=class_dl_ratios,
               label='Physics Prior', color='#2ecc71', alpha=0.8)
        plt.ylabel('Ratio (%)', fontsize=12)
        plt.title('Stacked Ratio by Class', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim([0, 100])

        # 6. 数值表格
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        table_data = []
        table_data.append(['Class', 'Gate', 'DL%', 'Physics%'])
        for i, cls_name in enumerate(class_names):
            mean_gate = np.mean(class_gates[i])
            dl = mean_gate * 100
            ph = (1 - mean_gate) * 100
            table_data.append([cls_name, f'{mean_gate:.3f}', f'{dl:.1f}', f'{ph:.1f}'])
        table_data.append(['Average', f'{avg_gate:.3f}', f'{dl_ratio:.1f}', f'{physics_ratio:.1f}'])

        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 设置表头样式
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 设置最后一行（平均值）样式
        for i in range(4):
            table[(len(table_data)-1, i)].set_facecolor('#ecf0f1')
            table[(len(table_data)-1, i)].set_text_props(weight='bold')

        plt.title('Gate Statistics Table', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gate_analysis.png'), dpi=300, bbox_inches='tight')
        print(f'Gate分析已保存: {os.path.join(save_dir, "gate_analysis.png")}')
    
    print('\n' + '='*80)
    print('测试完成！')
    print('='*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='轻量级CCGA模型测试')
    
    parser.add_argument('--dataset', type=str, default='tongyi_weidu_10x', help='数据集目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--backbone', type=str, default='simple_cnn', help='骨干网络')
    parser.add_argument('--feature_dim', type=int, default=256, help='特征维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--save_results', action='store_true', help='保存结果图表')
    
    args = parser.parse_args()
    
    test(args)

