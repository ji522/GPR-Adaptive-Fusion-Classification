"""
GPR数据集加载器
支持加载VV图像、VH图像和相干矩阵
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as sio
from sklearn.model_selection import train_test_split


class GPRDataset(Dataset):
    """GPR数据集"""
    
    def __init__(self, dataset_dir, split='train', train_ratio=0.8, random_seed=42, transform=None):
        """
        Args:
            dataset_dir: 数据集目录
            split: 'train' or 'test'
            train_ratio: 训练集比例
            random_seed: 随机种子
            transform: 数据增强（可选）
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        
        # 类别映射
        self.classes = ['PVC', '二面角', '含水', '多分支', '电缆', '空洞', '金属']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 加载数据列表
        self.samples = []
        self._load_samples(train_ratio, random_seed)
        
    def _load_samples(self, train_ratio, random_seed):
        """加载样本列表"""
        all_samples = []
        
        for cls in self.classes:
            cls_dir = os.path.join(self.dataset_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            
            # 获取所有VV矩阵文件
            vv_mat_dir = os.path.join(cls_dir, 'VV_matrices')
            mat_files = sorted([f for f in os.listdir(vv_mat_dir) if f.endswith('.mat')])
            
            for mat_file in mat_files:
                base_name = mat_file.replace('.mat', '')
                
                sample = {
                    'class': cls,
                    'label': self.class_to_idx[cls],
                    'base_name': base_name,
                    'vv_img': os.path.join(cls_dir, 'VV_images', base_name + '.jpg'),
                    'vh_img': os.path.join(cls_dir, 'VH_images', base_name + '.jpg'),
                    'vv_mat': os.path.join(cls_dir, 'VV_matrices', mat_file),
                    'vh_mat': os.path.join(cls_dir, 'VH_matrices', mat_file),
                    'coherence': os.path.join(cls_dir, 'VV_VH_matrices', mat_file)
                }
                
                all_samples.append(sample)
        
        # 划分训练集和测试集
        if train_ratio < 1.0:
            train_samples, test_samples = train_test_split(
                all_samples,
                train_size=train_ratio,
                random_state=random_seed,
                stratify=[s['label'] for s in all_samples]
            )
            
            if self.split == 'train':
                self.samples = train_samples
            else:
                self.samples = test_samples
        else:
            self.samples = all_samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        vv_img = Image.open(sample['vv_img']).convert('L')  # 灰度图
        vh_img = Image.open(sample['vh_img']).convert('L')
        
        # 转换为tensor
        vv_img = torch.from_numpy(np.array(vv_img)).float().unsqueeze(0) / 255.0
        vh_img = torch.from_numpy(np.array(vh_img)).float().unsqueeze(0) / 255.0
        
        # 加载相干矩阵
        coherence = sio.loadmat(sample['coherence'])['data1']
        coherence = torch.from_numpy(coherence).float().unsqueeze(0)
        
        # 归一化相干矩阵
        coherence = (coherence - coherence.min()) / (coherence.max() - coherence.min() + 1e-8)
        
        # 标签
        label = sample['label']
        
        return vv_img, vh_img, coherence, label


def create_dataloaders(dataset_dir, batch_size=16, train_ratio=0.8, num_workers=4, random_seed=42):
    """
    创建训练和测试数据加载器
    Args:
        dataset_dir: 数据集目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        num_workers: 数据加载线程数
        random_seed: 随机种子
    Returns:
        train_loader, test_loader
    """
    # 创建数据集
    train_dataset = GPRDataset(
        dataset_dir=dataset_dir,
        split='train',
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    test_dataset = GPRDataset(
        dataset_dir=dataset_dir,
        split='test',
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # 测试数据加载器
    print('='*80)
    print('数据加载器测试')
    print('='*80)
    
    dataset_dir = 'tongyi_weidu_10x'
    
    train_loader, test_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=8,
        train_ratio=0.8,
        num_workers=0
    )
    
    print(f'\n数据集: {dataset_dir}')
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'测试集大小: {len(test_loader.dataset)}')
    print(f'批次数（训练）: {len(train_loader)}')
    print(f'批次数（测试）: {len(test_loader)}')
    
    # 测试加载一个批次
    vv_img, vh_img, coherence, label = next(iter(train_loader))
    
    print(f'\n批次数据形状:')
    print(f'  VV图像: {vv_img.shape}')
    print(f'  VH图像: {vh_img.shape}')
    print(f'  相干矩阵: {coherence.shape}')
    print(f'  标签: {label.shape}')
    print(f'  标签值: {label.tolist()}')
    
    print('\n' + '='*80)
    print('✅ 数据加载器测试通过！')
    print('='*80)

