"""生成数据增强报告"""

import os
import scipy.io as sio
from PIL import Image
import numpy as np
from datetime import datetime

def generate_report(dataset_dir='tongyi_weidu_5x', output_file='augmentation_report.txt'):
    """生成数据增强报告"""
    
    classes = ['PVC', '二面角', '含水', '多分支', '电缆', '空洞', '金属']
    
    report = []
    report.append('='*80)
    report.append('GPR数据集增强报告')
    report.append('='*80)
    report.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report.append(f'数据集目录: {dataset_dir}')
    report.append('='*80)
    
    # 统计每个类别
    total_samples = 0
    class_stats = []
    
    for cls in classes:
        cls_path = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_path):
            continue
        
        vv_mats = [f for f in os.listdir(os.path.join(cls_path, 'VV_matrices')) if f.endswith('.mat')]
        original = [f for f in vv_mats if 'aug' not in f]
        augmented = [f for f in vv_mats if 'aug' in f]
        
        class_stats.append({
            'class': cls,
            'original': len(original),
            'augmented': len(augmented),
            'total': len(vv_mats)
        })
        
        total_samples += len(vv_mats)
    
    # 写入类别统计
    report.append('\n【类别统计】')
    report.append(f'{"类别":<10} {"原始样本":<12} {"增强样本":<12} {"总样本":<12}')
    report.append('-'*50)
    
    for stat in class_stats:
        report.append(f'{stat["class"]:<10} {stat["original"]:<12} {stat["augmented"]:<12} {stat["total"]:<12}')
    
    report.append('-'*50)
    total_original = sum(s['original'] for s in class_stats)
    total_augmented = sum(s['augmented'] for s in class_stats)
    report.append(f'{"总计":<10} {total_original:<12} {total_augmented:<12} {total_samples:<12}')
    
    # 增强倍数
    aug_factor = total_samples / total_original if total_original > 0 else 0
    report.append(f'\n增强倍数: {aug_factor:.1f}x')
    
    # 检查数据质量
    report.append('\n' + '='*80)
    report.append('【数据质量检查】')
    report.append('='*80)
    
    # 检查第一个类别的第一个样本
    first_cls = class_stats[0]['class']
    cls_path = os.path.join(dataset_dir, first_cls)
    
    # 找到第一个原始样本和第一个增强样本
    vv_mat_dir = os.path.join(cls_path, 'VV_matrices')
    all_files = sorted(os.listdir(vv_mat_dir))
    original_file = [f for f in all_files if 'aug' not in f][0]
    augmented_file = [f for f in all_files if 'aug' in f][0]
    
    # 加载并检查
    orig_vv = sio.loadmat(os.path.join(vv_mat_dir, original_file))['data1']
    aug_vv = sio.loadmat(os.path.join(vv_mat_dir, augmented_file))['data1']
    
    report.append(f'\n样本检查（{first_cls}类）:')
    report.append(f'  原始样本: {original_file}')
    report.append(f'    - 形状: {orig_vv.shape}')
    report.append(f'    - 类型: {orig_vv.dtype}')
    report.append(f'    - 是否复数: {np.iscomplexobj(orig_vv)}')
    
    report.append(f'\n  增强样本: {augmented_file}')
    report.append(f'    - 形状: {aug_vv.shape}')
    report.append(f'    - 类型: {aug_vv.dtype}')
    report.append(f'    - 是否复数: {np.iscomplexobj(aug_vv)}')
    
    # 检查相干矩阵
    vv_vh_mat_dir = os.path.join(cls_path, 'VV_VH_matrices')
    orig_vv_vh = sio.loadmat(os.path.join(vv_vh_mat_dir, original_file))['data1']
    aug_vv_vh = sio.loadmat(os.path.join(vv_vh_mat_dir, augmented_file))['data1']
    
    report.append(f'\n相干矩阵检查:')
    report.append(f'  原始相干矩阵:')
    report.append(f'    - 形状: {orig_vv_vh.shape}')
    report.append(f'    - 类型: {orig_vv_vh.dtype}')
    report.append(f'    - 是否复数: {np.iscomplexobj(orig_vv_vh)}')
    report.append(f'    - 数值范围: [{orig_vv_vh.min():.6f}, {orig_vv_vh.max():.6f}]')
    
    report.append(f'\n  增强相干矩阵:')
    report.append(f'    - 形状: {aug_vv_vh.shape}')
    report.append(f'    - 类型: {aug_vv_vh.dtype}')
    report.append(f'    - 是否复数: {np.iscomplexobj(aug_vv_vh)}')
    report.append(f'    - 数值范围: [{aug_vv_vh.min():.6f}, {aug_vv_vh.max():.6f}]')
    
    # 增强方法统计
    report.append('\n' + '='*80)
    report.append('【增强方法统计】')
    report.append('='*80)
    
    aug_methods = {}
    for f in all_files:
        if 'aug' in f:
            method = f.split('_')[-1].replace('.mat', '')
            aug_methods[method] = aug_methods.get(method, 0) + 1
    
    report.append(f'\n{"增强方法":<15} {"样本数":<10}')
    report.append('-'*30)
    for method, count in sorted(aug_methods.items()):
        report.append(f'{method:<15} {count:<10}')
    
    report.append('\n' + '='*80)
    report.append('报告生成完成')
    report.append('='*80)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # 同时打印到控制台
    print('\n'.join(report))
    print(f'\n报告已保存到: {output_file}')

if __name__ == '__main__':
    generate_report('tongyi_weidu_5x', 'tongyi_weidu_5x/augmentation_report.txt')

