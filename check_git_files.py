"""
检查哪些文件会被 Git 追踪（上传到 GitHub）
"""

import os
import subprocess
from pathlib import Path

def get_file_size(filepath):
    """获取文件大小（MB）"""
    try:
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except:
        return 0

def check_git_files():
    """检查将要被 Git 追踪的文件"""
    
    print("=" * 80)
    print("检查将要上传到 GitHub 的文件")
    print("=" * 80)
    
    # 检查是否已经初始化 Git
    if not os.path.exists('.git'):
        print("\n⚠️  Git 仓库尚未初始化")
        print("请先运行: git init")
        return
    
    # 获取所有文件
    try:
        # 模拟 git add . 后会追踪的文件
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            print("❌ 无法获取 Git 状态")
            return
        
        lines = result.stdout.strip().split('\n')
        
        total_size = 0
        large_files = []
        file_count = 0
        
        print("\n📁 将要追踪的文件：\n")
        
        for line in lines:
            if not line.strip():
                continue
            
            # 解析文件路径
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            status = parts[0]
            filepath = parts[1]
            
            # 跳过删除的文件
            if status.startswith('D'):
                continue
            
            if os.path.isfile(filepath):
                size_mb = get_file_size(filepath)
                total_size += size_mb
                file_count += 1
                
                # 标记大文件
                if size_mb > 10:
                    large_files.append((filepath, size_mb))
                    print(f"  ⚠️  {filepath:<60} {size_mb:>8.2f} MB (大文件)")
                elif size_mb > 1:
                    print(f"  📄 {filepath:<60} {size_mb:>8.2f} MB")
                else:
                    print(f"  📄 {filepath:<60} {size_mb:>8.3f} MB")
        
        print("\n" + "=" * 80)
        print(f"📊 统计信息：")
        print(f"  文件总数: {file_count}")
        print(f"  总大小: {total_size:.2f} MB")
        
        if large_files:
            print(f"\n⚠️  发现 {len(large_files)} 个大文件（>10MB）：")
            for filepath, size in large_files:
                print(f"  - {filepath}: {size:.2f} MB")
            
            if any(size > 100 for _, size in large_files):
                print("\n❌ 警告：有文件超过 100MB，GitHub 会拒绝推送！")
                print("   建议：")
                print("   1. 将大文件添加到 .gitignore")
                print("   2. 或使用 Git LFS: git lfs track '*.pth'")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"❌ 错误: {e}")

def check_gitignore():
    """检查 .gitignore 配置"""
    print("\n" + "=" * 80)
    print("检查 .gitignore 配置")
    print("=" * 80)
    
    if not os.path.exists('.gitignore'):
        print("\n⚠️  .gitignore 文件不存在")
        print("建议创建 .gitignore 文件排除不需要的文件")
        return
    
    with open('.gitignore', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n✅ .gitignore 已配置，共 {len(lines)} 行规则")
    print("\n主要排除的内容：")
    
    important_patterns = [
        '__pycache__',
        '*.pth',
        '*.mat',
        'tongyi_weidu',
        'experiments',
        'neural_network_dataset'
    ]
    
    for pattern in important_patterns:
        if any(pattern in line for line in lines):
            print(f"  ✓ {pattern}")
        else:
            print(f"  ✗ {pattern} (未配置)")

def main():
    """主函数"""
    os.chdir(Path(__file__).parent)
    
    check_gitignore()
    check_git_files()
    
    print("\n💡 提示：")
    print("  1. 如果还没初始化 Git，运行: git init")
    print("  2. 添加文件到暂存区: git add .")
    print("  3. 再次运行本脚本查看将要提交的文件")
    print("  4. 提交: git commit -m '你的提交信息'")
    print("  5. 推送到 GitHub: git push -u origin main")
    print("\n详细步骤请查看: GitHub上传指南.md")
    print("=" * 80)

if __name__ == '__main__':
    main()

