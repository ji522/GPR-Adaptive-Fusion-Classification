# GitHub 上传脚本（PowerShell）
# 使用方法：在 PowerShell 中运行 .\upload_to_github.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "GitHub 上传助手" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否安装了 Git
try {
    $gitVersion = git --version
    Write-Host "✓ Git 已安装: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git 未安装，请先安装 Git: https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "步骤 1: 配置 Git 用户信息" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 检查 Git 配置
$userName = git config --global user.name
$userEmail = git config --global user.email

if ([string]::IsNullOrEmpty($userName) -or [string]::IsNullOrEmpty($userEmail)) {
    Write-Host "需要配置 Git 用户信息" -ForegroundColor Yellow
    
    $name = Read-Host "请输入你的名字（会显示在提交记录中）"
    $email = Read-Host "请输入你的邮箱"
    
    git config --global user.name "$name"
    git config --global user.email "$email"
    
    Write-Host "✓ Git 用户信息已配置" -ForegroundColor Green
} else {
    Write-Host "✓ Git 用户信息已配置:" -ForegroundColor Green
    Write-Host "  用户名: $userName" -ForegroundColor Gray
    Write-Host "  邮箱: $userEmail" -ForegroundColor Gray
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "步骤 2: 初始化 Git 仓库" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

if (Test-Path ".git") {
    Write-Host "✓ Git 仓库已存在" -ForegroundColor Green
} else {
    Write-Host "初始化 Git 仓库..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git 仓库初始化完成" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "步骤 3: 检查将要上传的文件" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "运行文件检查脚本..." -ForegroundColor Yellow
python check_git_files.py

Write-Host ""
$continue = Read-Host "是否继续添加文件到 Git？(y/n)"
if ($continue -ne 'y') {
    Write-Host "已取消" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "步骤 4: 添加文件到暂存区" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "添加所有文件..." -ForegroundColor Yellow
git add .

Write-Host "✓ 文件已添加到暂存区" -ForegroundColor Green
Write-Host ""
Write-Host "将要提交的文件：" -ForegroundColor Cyan
git status --short

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "步骤 5: 提交到本地仓库" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$commitMessage = Read-Host "请输入提交信息（默认：Initial commit）"
if ([string]::IsNullOrEmpty($commitMessage)) {
    $commitMessage = "Initial commit: 基于物理先验与深度学习自适应融合的GPR目标分类方法"
}

git commit -m "$commitMessage"
Write-Host "✓ 已提交到本地仓库" -ForegroundColor Green

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "步骤 6: 连接到 GitHub 远程仓库" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "请先在 GitHub 上创建一个新仓库：" -ForegroundColor Yellow
Write-Host "  1. 访问 https://github.com/new" -ForegroundColor Gray
Write-Host "  2. 填写仓库名称（例如：GPR-Adaptive-Fusion-Classification）" -ForegroundColor Gray
Write-Host "  3. 选择 Public 或 Private" -ForegroundColor Gray
Write-Host "  4. 不要勾选 'Initialize this repository with a README'" -ForegroundColor Gray
Write-Host "  5. 点击 'Create repository'" -ForegroundColor Gray
Write-Host ""

$repoUrl = Read-Host "请输入你的 GitHub 仓库 URL（例如：https://github.com/username/repo.git）"

if ([string]::IsNullOrEmpty($repoUrl)) {
    Write-Host "✗ 未输入仓库 URL，已取消" -ForegroundColor Red
    exit 1
}

# 检查是否已经添加了 remote
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "远程仓库已存在: $existingRemote" -ForegroundColor Yellow
    $updateRemote = Read-Host "是否更新为新的 URL？(y/n)"
    if ($updateRemote -eq 'y') {
        git remote set-url origin $repoUrl
        Write-Host "✓ 远程仓库 URL 已更新" -ForegroundColor Green
    }
} else {
    git remote add origin $repoUrl
    Write-Host "✓ 远程仓库已添加" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "步骤 7: 推送到 GitHub" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "准备推送到 GitHub..." -ForegroundColor Yellow
Write-Host "注意：首次推送可能需要输入 GitHub 用户名和 Personal Access Token" -ForegroundColor Yellow
Write-Host ""

$push = Read-Host "是否现在推送到 GitHub？(y/n)"
if ($push -eq 'y') {
    # 尝试推送到 main 分支
    git push -u origin main 2>$null
    if ($LASTEXITCODE -ne 0) {
        # 如果 main 不存在，尝试 master
        Write-Host "尝试推送到 master 分支..." -ForegroundColor Yellow
        git branch -M main
        git push -u origin main
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================================================================" -ForegroundColor Green
        Write-Host "✓ 成功上传到 GitHub！" -ForegroundColor Green
        Write-Host "================================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "你的仓库地址: $repoUrl" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "✗ 推送失败" -ForegroundColor Red
        Write-Host "可能的原因：" -ForegroundColor Yellow
        Write-Host "  1. 认证失败 - 需要使用 Personal Access Token 而不是密码" -ForegroundColor Gray
        Write-Host "  2. 文件太大 - 检查是否有超过 100MB 的文件" -ForegroundColor Gray
        Write-Host "  3. 网络问题 - 检查网络连接" -ForegroundColor Gray
        Write-Host ""
        Write-Host "详细帮助请查看: GitHub上传指南.md" -ForegroundColor Cyan
    }
} else {
    Write-Host ""
    Write-Host "已取消推送，你可以稍后手动推送：" -ForegroundColor Yellow
    Write-Host "  git push -u origin main" -ForegroundColor Gray
}

Write-Host ""
Write-Host "完成！" -ForegroundColor Green

