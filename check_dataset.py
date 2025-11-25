"""检查 tongyi_weidu 数据集详细信息"""

import os
import scipy.io as sio
from PIL import Image
import numpy as np

print('='*80)
print('tongyi_weidu 数据集详细信息')
print('='*80)

classes = ['PVC', '二面角', '含水', '多分支', '电缆', '空洞', '金属']

total_samples = 0
for cls in classes:
    cls_path = os.path.join('tongyi_weidu', cls)
    if os.path.exists(cls_path):
        # 统计各类数据
        vv_imgs = len([f for f in os.listdir(os.path.join(cls_path, 'VV_images')) if f.endswith('.png')])
        vh_imgs = len([f for f in os.listdir(os.path.join(cls_path, 'VH_images')) if f.endswith('.png')])
        vv_mats = len([f for f in os.listdir(os.path.join(cls_path, 'VV_matrices')) if f.endswith('.mat')])
        vh_mats = len([f for f in os.listdir(os.path.join(cls_path, 'VH_matrices')) if f.endswith('.mat')])
        vv_vh_mats = len([f for f in os.listdir(os.path.join(cls_path, 'VV_VH_matrices')) if f.endswith('.mat')])
        
        print(f'\n【{cls}】')
        print(f'  VV图像: {vv_imgs} 张')
        print(f'  VH图像: {vh_imgs} 张')
        print(f'  VV矩阵: {vv_mats} 个')
        print(f'  VH矩阵: {vh_mats} 个')
        print(f'  VV_VH相干矩阵: {vv_vh_mats} 个')
        
        total_samples += vv_imgs

print(f'\n总样本数: {total_samples}')

# 检查一个样本的详细信息
print('\n' + '='*80)
print('样本详细信息（以PVC为例）')
print('='*80)

# 检查VV_images目录下的文件
vv_img_dir = 'tongyi_weidu/PVC/VV_images'
if os.path.exists(vv_img_dir):
    files = os.listdir(vv_img_dir)
    print(f'\nVV_images目录下的文件: {files[:5] if len(files) > 5 else files}')

# 检查矩阵文件
sample_vv = sio.loadmat('tongyi_weidu/PVC/VV_matrices/PVC_1.mat')
sample_vh = sio.loadmat('tongyi_weidu/PVC/VH_matrices/PVC_1.mat')
sample_vv_vh = sio.loadmat('tongyi_weidu/PVC/VV_VH_matrices/PVC_1.mat')

print(f'\nVV矩阵形状: {sample_vv["data1"].shape}')
print(f'VV矩阵类型: {sample_vv["data1"].dtype}')
print(f'VV矩阵是否复数: {np.iscomplexobj(sample_vv["data1"])}')
print(f'\nVH矩阵形状: {sample_vh["data1"].shape}')
print(f'VH矩阵类型: {sample_vh["data1"].dtype}')
print(f'VH矩阵是否复数: {np.iscomplexobj(sample_vh["data1"])}')
print(f'\nVV_VH矩阵形状: {sample_vv_vh["data1"].shape}')
print(f'VV_VH矩阵类型: {sample_vv_vh["data1"].dtype}')
print(f'VV_VH矩阵是否复数: {np.iscomplexobj(sample_vv_vh["data1"])}')

print('\n' + '='*80)
print('数据扩充建议')
print('='*80)
print('\n可用的数据增强方法：')
print('  1. 水平翻转 (Horizontal Flip)')
print('  2. 垂直翻转 (Vertical Flip)')
print('  3. 旋转 (Rotation: 90°, 180°, 270°)')
print('  4. 随机裁剪后resize (Random Crop + Resize)')
print('  5. 添加高斯噪声 (Gaussian Noise)')
print('  6. 亮度调整 (Brightness Adjustment)')
print('  7. 对比度调整 (Contrast Adjustment)')
print('  8. 弹性变换 (Elastic Transform)')
print('\n注意：')
print('  - 对于复数矩阵，需要同时处理实部和虚部')
print('  - VV和VH矩阵需要同步增强')
print('  - 增强后需要重新计算VV_VH相干矩阵')
print('='*80)

