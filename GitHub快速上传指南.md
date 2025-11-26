# GitHub 快速上传指南

## 🚀 首次上传完整流程

### 第一步：初始化 Git 仓库
```powershell
git init
```

### 第二步：配置用户信息（首次使用需要）
```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

### 第三步：添加文件到暂存区
```powershell
git add .
```

### 第四步：提交到本地仓库
```powershell
git commit -m "Initial commit: 项目描述"
```

### 第五步：在 GitHub 创建远程仓库
1. 访问 https://github.com/new
2. 填写仓库名称（建议用英文）
3. 选择 Public（公开）或 Private（私有）
4. **不要**勾选 "Add a README file"
5. 点击 "Create repository"

### 第六步：连接远程仓库
```powershell
git remote add origin https://github.com/你的用户名/仓库名.git
```

### 第七步：推送到 GitHub
```powershell
git branch -M main
git push -u origin main
```

---

## 🔧 常见问题解决方案

### 问题 1：网络连接失败（使用 Clash 代理）
```powershell
# 设置 Git 使用代理（端口根据你的 Clash 设置）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

### 问题 2：SSH 连接失败（推荐使用 SSH）

#### 1. 生成 SSH 密钥
```powershell
ssh-keygen -t ed25519 -C "你的邮箱"
```
一路按回车（3次）

#### 2. 查看并复制公钥
```powershell
cat ~/.ssh/id_ed25519.pub
```

#### 3. 添加到 GitHub
- 访问：https://github.com/settings/keys
- 点击 "New SSH key"
- 粘贴公钥，点击 "Add SSH key"

#### 4. 配置 SSH 使用 443 端口（解决端口被封问题）
```powershell
# 创建 SSH 配置文件
@"
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
"@ | Out-File -FilePath $HOME\.ssh\config -Encoding ASCII

# 修复权限
icacls $HOME\.ssh\config /inheritance:r
icacls $HOME\.ssh\config /grant:r "$($env:USERNAME):(R,W)"
```

#### 5. 测试 SSH 连接
```powershell
ssh -T git@github.com
```
成功会显示：`Hi 你的用户名! You've successfully authenticated...`

#### 6. 修改远程仓库地址为 SSH
```powershell
git remote set-url origin git@github.com:你的用户名/仓库名.git
```

#### 7. 推送
```powershell
git push -u origin main
```

---

## 📝 后续更新代码（三步走）

```powershell
# 1. 添加修改的文件
git add .

# 2. 提交修改
git commit -m "描述你做了什么修改"

# 3. 推送到 GitHub
git push
```

---

## 🔍 常用 Git 命令

```powershell
# 查看当前状态
git status

# 查看提交历史
git log

# 查看远程仓库地址
git remote -v

# 拉取远程更新
git pull

# 查看配置
git config --list
```

---

## 💡 最佳实践

1. **提交信息要清晰**：描述你做了什么修改
2. **经常提交**：不要等到改了很多才提交
3. **使用 .gitignore**：避免上传不必要的文件
4. **使用 SSH**：比 HTTPS 更稳定，不需要每次输入密码

---

## 📚 参考链接

- GitHub 官方文档：https://docs.github.com/
- Git 官方文档：https://git-scm.com/doc
- SSH 密钥设置：https://github.com/settings/keys

---

**创建时间**：2025-11-25  
**适用环境**：Windows PowerShell + Clash 代理

