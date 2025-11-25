"""
tongyi_weidu 数据集扩充脚本
对GPR B-scan图像和极化相干矩阵进行数据增强
"""

import os
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
from tqdm import tqdm
import argparse

class GPRDataAugmentor:
    """GPR数据增强器"""
    
    def __init__(self, augmentation_factor=5):
        """
        初始化
        Args:
            augmentation_factor: 增强倍数（每个原始样本生成多少个增强样本）
        """
        self.aug_factor = augmentation_factor
        
    def augment_complex_matrix(self, matrix, aug_type):
        """
        增强复数矩阵
        Args:
            matrix: 复数矩阵 (H, W)
            aug_type: 增强类型
        Returns:
            增强后的复数矩阵
        """
        if aug_type == 'hflip':
            # 水平翻转
            return np.fliplr(matrix)
        
        elif aug_type == 'vflip':
            # 垂直翻转
            return np.flipud(matrix)
        
        elif aug_type == 'noise_weak':
            # 添加弱高斯噪声
            noise_level = 0.01
            real_noise = np.random.normal(0, noise_level, matrix.shape)
            imag_noise = np.random.normal(0, noise_level, matrix.shape)
            noise = real_noise + 1j * imag_noise
            return matrix + noise

        elif aug_type == 'noise_medium':
            # 添加中等高斯噪声
            noise_level = 0.02
            real_noise = np.random.normal(0, noise_level, matrix.shape)
            imag_noise = np.random.normal(0, noise_level, matrix.shape)
            noise = real_noise + 1j * imag_noise
            return matrix + noise

        elif aug_type == 'noise_strong':
            # 添加强高斯噪声
            noise_level = 0.03
            real_noise = np.random.normal(0, noise_level, matrix.shape)
            imag_noise = np.random.normal(0, noise_level, matrix.shape)
            noise = real_noise + 1j * imag_noise
            return matrix + noise

        elif aug_type == 'brightness':
            # 亮度调整（缩放幅度）
            scale = np.random.uniform(0.9, 1.1)
            return matrix * scale
        
        else:
            return matrix
    
    def augment_image(self, image, aug_type):
        """
        增强图像
        Args:
            image: PIL Image对象
            aug_type: 增强类型
        Returns:
            增强后的PIL Image对象
        """
        img_array = np.array(image)
        
        if aug_type == 'hflip':
            img_array = np.fliplr(img_array)
        elif aug_type == 'vflip':
            img_array = np.flipud(img_array)
        elif aug_type == 'noise_weak':
            # 添加弱高斯噪声
            noise = np.random.normal(0, 3, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        elif aug_type == 'noise_medium':
            # 添加中等高斯噪声
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        elif aug_type == 'noise_strong':
            # 添加强高斯噪声
            noise = np.random.normal(0, 8, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        elif aug_type == 'brightness':
            # 亮度调整
            scale = np.random.uniform(0.9, 1.1)
            img_array = np.clip(img_array * scale, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def calculate_coherence(self, vv_matrix, vh_matrix):
        """
        计算VV和VH的交叉相干系数
        Args:
            vv_matrix: VV复数矩阵
            vh_matrix: VH复数矩阵
        Returns:
            交叉相干系数矩阵（实数）
        """
        return np.abs(vv_matrix * np.conj(vh_matrix))
    
    def augment_sample(self, vv_mat, vh_mat, vv_img, vh_img, aug_types):
        """
        增强单个样本
        Args:
            vv_mat: VV矩阵
            vh_mat: VH矩阵
            vv_img: VV图像
            vh_img: VH图像
            aug_types: 增强类型列表
        Returns:
            增强后的数据列表
        """
        augmented_data = []
        
        for aug_type in aug_types:
            # 增强矩阵
            aug_vv_mat = self.augment_complex_matrix(vv_mat, aug_type)
            aug_vh_mat = self.augment_complex_matrix(vh_mat, aug_type)
            
            # 计算新的相干矩阵
            aug_vv_vh_mat = self.calculate_coherence(aug_vv_mat, aug_vh_mat)
            
            # 增强图像
            aug_vv_img = self.augment_image(vv_img, aug_type)
            aug_vh_img = self.augment_image(vh_img, aug_type)
            
            augmented_data.append({
                'vv_mat': aug_vv_mat,
                'vh_mat': aug_vh_mat,
                'vv_vh_mat': aug_vv_vh_mat,
                'vv_img': aug_vv_img,
                'vh_img': aug_vh_img,
                'aug_type': aug_type
            })
        
        return augmented_data
    
    def get_augmentation_types(self):
        """获取增强类型列表（不包含旋转和平移）"""
        all_types = ['hflip', 'vflip', 'noise_weak', 'noise_medium', 'noise_strong', 'brightness']
        return all_types[:self.aug_factor]


def augment_dataset(input_dir='tongyi_weidu', output_dir='tongyi_weidu_augmented', aug_factor=5):
    """
    增强整个数据集
    Args:
        input_dir: 输入数据集目录
        output_dir: 输出数据集目录
        aug_factor: 增强倍数
    """
    print('='*80)
    print(f'GPR数据集增强')
    print('='*80)
    print(f'输入目录: {input_dir}')
    print(f'输出目录: {output_dir}')
    print(f'增强倍数: {aug_factor}x')
    print('='*80)

    # 创建增强器
    augmentor = GPRDataAugmentor(augmentation_factor=aug_factor)
    aug_types = augmentor.get_augmentation_types()
    print(f'\n增强方法: {aug_types}')

    # 类别列表
    classes = ['PVC', '二面角', '含水', '多分支', '电缆', '空洞', '金属']

    # 统计信息
    total_original = 0
    total_augmented = 0

    # 遍历每个类别
    for cls in classes:
        print(f'\n处理类别: {cls}')

        cls_input_dir = os.path.join(input_dir, cls)
        cls_output_dir = os.path.join(output_dir, cls)

        if not os.path.exists(cls_input_dir):
            print(f'  跳过（目录不存在）')
            continue

        # 创建输出目录
        for subdir in ['VV_images', 'VH_images', 'VV_matrices', 'VH_matrices', 'VV_VH_matrices']:
            os.makedirs(os.path.join(cls_output_dir, subdir), exist_ok=True)

        # 获取所有样本文件
        vv_mat_dir = os.path.join(cls_input_dir, 'VV_matrices')
        mat_files = sorted([f for f in os.listdir(vv_mat_dir) if f.endswith('.mat')])

        print(f'  原始样本数: {len(mat_files)}')
        total_original += len(mat_files)

        # 处理每个样本
        for mat_file in tqdm(mat_files, desc=f'  增强{cls}'):
            base_name = mat_file.replace('.mat', '')

            # 加载原始数据
            vv_mat_path = os.path.join(cls_input_dir, 'VV_matrices', mat_file)
            vh_mat_path = os.path.join(cls_input_dir, 'VH_matrices', mat_file)
            vv_img_path = os.path.join(cls_input_dir, 'VV_images', base_name + '.jpg')
            vh_img_path = os.path.join(cls_input_dir, 'VH_images', base_name + '.jpg')

            vv_mat = sio.loadmat(vv_mat_path)['data1']
            vh_mat = sio.loadmat(vh_mat_path)['data1']
            vv_img = Image.open(vv_img_path)
            vh_img = Image.open(vh_img_path)

            # 保存原始样本
            sio.savemat(os.path.join(cls_output_dir, 'VV_matrices', mat_file), {'data1': vv_mat})
            sio.savemat(os.path.join(cls_output_dir, 'VH_matrices', mat_file), {'data1': vh_mat})
            vv_vh_mat = sio.loadmat(os.path.join(cls_input_dir, 'VV_VH_matrices', mat_file))['data1']
            sio.savemat(os.path.join(cls_output_dir, 'VV_VH_matrices', mat_file), {'data1': vv_vh_mat})
            vv_img.save(os.path.join(cls_output_dir, 'VV_images', base_name + '.jpg'))
            vh_img.save(os.path.join(cls_output_dir, 'VH_images', base_name + '.jpg'))

            total_augmented += 1

            # 生成增强样本
            augmented_samples = augmentor.augment_sample(vv_mat, vh_mat, vv_img, vh_img, aug_types)

            for idx, aug_data in enumerate(augmented_samples):
                aug_name = f"{base_name}_aug{idx+1}_{aug_data['aug_type']}"

                # 保存增强后的矩阵
                sio.savemat(os.path.join(cls_output_dir, 'VV_matrices', aug_name + '.mat'),
                           {'data1': aug_data['vv_mat']})
                sio.savemat(os.path.join(cls_output_dir, 'VH_matrices', aug_name + '.mat'),
                           {'data1': aug_data['vh_mat']})
                sio.savemat(os.path.join(cls_output_dir, 'VV_VH_matrices', aug_name + '.mat'),
                           {'data1': aug_data['vv_vh_mat']})

                # 保存增强后的图像
                aug_data['vv_img'].save(os.path.join(cls_output_dir, 'VV_images', aug_name + '.jpg'))
                aug_data['vh_img'].save(os.path.join(cls_output_dir, 'VH_images', aug_name + '.jpg'))

                total_augmented += 1

        print(f'  增强后样本数: {len(mat_files) * (1 + aug_factor)}')

    print('\n' + '='*80)
    print('增强完成！')
    print('='*80)
    print(f'原始样本总数: {total_original}')
    print(f'增强后样本总数: {total_augmented}')
    print(f'增强倍数: {total_augmented / total_original:.1f}x')
    print('='*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPR数据集增强')
    parser.add_argument('--input', type=str, default='tongyi_weidu', help='输入数据集目录')
    parser.add_argument('--output', type=str, default='tongyi_weidu_augmented', help='输出数据集目录')
    parser.add_argument('--factor', type=int, default=5, help='增强倍数')

    args = parser.parse_args()

    augment_dataset(args.input, args.output, args.factor)

