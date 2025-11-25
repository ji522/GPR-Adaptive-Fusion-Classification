# GitHub 上传完整指南

## 📋 准备工作

### 1. 安装 Git
如果还没有安装 Git，请先安装：
- Windows: 下载 [Git for Windows](https://git-scm.com/download/win)
- 安装时选择默认选项即可

### 2. 配置 Git（首次使用）
```powershell
# 设置你的用户名和邮箱（会显示在提交记录中）
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

### 3. 创建 GitHub 账号
- 访问 [GitHub](https://github.com) 注册账号
- 验证邮箱

---

## 🚀 上传步骤

### 方法一：通过命令行上传（推荐）

#### 步骤 1: 初始化 Git 仓库
在项目根目录（`g:\jsy_dataset_and_model`）打开 PowerShell：

```powershell
# 进入项目目录
cd g:\jsy_dataset_and_model

# 初始化 Git 仓库
git init

# 查看当前状态
git status
```

#### 步骤 2: 添加文件到暂存区
```powershell
# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 查看将要提交的文件
git status
```

#### 步骤 3: 提交到本地仓库
```powershell
# 提交并添加说明
git commit -m "Initial commit: 基于物理先验与深度学习自适应融合的GPR目标分类方法"
```

#### 步骤 4: 在 GitHub 上创建远程仓库
1. 登录 GitHub
2. 点击右上角 "+" → "New repository"
3. 填写信息：
   - **Repository name**: `GPR-Adaptive-Fusion-Classification`（或你喜欢的名字）
   - **Description**: `基于物理先验与深度学习自适应融合的GPR目标分类方法`
   - **Public/Private**: 选择公开或私有
   - **不要勾选** "Initialize this repository with a README"（因为我们已经有了）
4. 点击 "Create repository"

#### 步骤 5: 连接远程仓库并推送
GitHub 会显示一些命令，你需要执行：

```powershell
# 添加远程仓库（替换成你的 GitHub 用户名和仓库名）
git remote add origin https://github.com/你的用户名/GPR-Adaptive-Fusion-Classification.git

# 推送到 GitHub（首次推送）
git push -u origin master
# 或者如果默认分支是 main：
git push -u origin main
```

**注意**：首次推送时会要求输入 GitHub 用户名和密码（或 Personal Access Token）

---

### 方法二：使用 GitHub Desktop（图形界面，更简单）

#### 步骤 1: 安装 GitHub Desktop
- 下载 [GitHub Desktop](https://desktop.github.com/)
- 安装并登录你的 GitHub 账号

#### 步骤 2: 添加本地仓库
1. 打开 GitHub Desktop
2. 点击 "File" → "Add local repository"
3. 选择 `g:\jsy_dataset_and_model` 目录
4. 如果提示"不是 Git 仓库"，点击 "create a repository"

#### 步骤 3: 提交更改
1. 在左侧会看到所有更改的文件
2. 在底部输入提交信息：`Initial commit: 基于物理先验与深度学习自适应融合的GPR目标分类方法`
3. 点击 "Commit to main"

#### 步骤 4: 发布到 GitHub
1. 点击顶部的 "Publish repository"
2. 填写仓库名称和描述
3. 选择公开或私有
4. 点击 "Publish repository"

---

## 📝 后续更新

### 修改文件后如何更新到 GitHub

```powershell
# 1. 查看修改了哪些文件
git status

# 2. 添加修改的文件
git add .

# 3. 提交修改
git commit -m "更新说明，例如：修复了门控饱和问题"

# 4. 推送到 GitHub
git push
```

---

## ⚠️ 重要提示

### 1. 关于大文件
- `.gitignore` 已经配置好，会自动排除：
  - 数据集文件夹（太大）
  - 模型文件 `.pth`（太大）
  - Python 缓存 `__pycache__`
  - 实验结果图片（部分保留）

### 2. 如果需要上传大文件
GitHub 单个文件限制 100MB，如果需要上传大文件：

**选项 1: 使用 Git LFS（Large File Storage）**
```powershell
# 安装 Git LFS
git lfs install

# 追踪大文件类型
git lfs track "*.pth"
git lfs track "*.mat"

# 添加 .gitattributes
git add .gitattributes

# 正常提交和推送
git add .
git commit -m "添加模型文件"
git push
```

**选项 2: 使用云存储**
- 将数据集和模型上传到：
  - Google Drive
  - 百度网盘
  - OneDrive
- 在 README 中添加下载链接

### 3. 如果推送失败
可能是因为文件太大，解决方法：

```powershell
# 查看哪些文件太大
git ls-files -s | awk '{if ($4 > 100000000) print $4, $2}'

# 从暂存区移除大文件
git rm --cached 文件路径

# 添加到 .gitignore
echo "文件路径" >> .gitignore

# 重新提交
git add .
git commit -m "移除大文件"
git push
```

---

## 🎯 推荐的 README 文件

我已经为你创建了 `README_GITHUB.md`，建议：

```powershell
# 用 GitHub 版本替换原来的 README
mv README.md README_OLD.md
mv README_GITHUB.md README.md

# 提交更新
git add .
git commit -m "更新 README 为 GitHub 版本"
git push
```

---

## 📚 常用 Git 命令速查

```powershell
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 拉取最新代码
git pull

# 创建新分支
git checkout -b 新分支名

# 切换分支
git checkout 分支名

# 合并分支
git merge 分支名

# 撤销修改（未提交）
git checkout -- 文件名

# 撤销提交（保留修改）
git reset --soft HEAD^

# 查看差异
git diff
```

---

## 🔗 有用的链接

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub 官方指南](https://docs.github.com/cn)
- [Git 简明指南](https://rogerdudler.github.io/git-guide/index.zh.html)
- [GitHub Desktop 文档](https://docs.github.com/cn/desktop)

---

## ❓ 常见问题

### Q1: 推送时要求输入密码，但密码不对？
A: GitHub 已经不支持密码登录，需要使用 Personal Access Token：
1. GitHub 网页 → Settings → Developer settings → Personal access tokens → Generate new token
2. 勾选 `repo` 权限
3. 复制生成的 token
4. 推送时用 token 代替密码

### Q2: 如何删除已经推送的敏感文件？
A: 使用 `git filter-branch` 或 BFG Repo-Cleaner，详见 [GitHub 文档](https://docs.github.com/cn/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)

### Q3: 如何让别人协作？
A: 
1. 仓库页面 → Settings → Collaborators
2. 添加协作者的 GitHub 用户名

---

**祝你上传顺利！如有问题随时问我。** 🎉

