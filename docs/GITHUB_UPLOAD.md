# GitHub 上传指南

## 上传前检查

仓库已经在本地初始化，文件也已经加入暂存区。上传前先确认Git状态和敏感文件检查结果：

```powershell
cd C:\Users\Lin\Desktop\fx\lexchain-toulmin-study
git status
git diff --cached --stat
git ls-files "*.jsonl" "*.docx" "*.pdf" ".env*"
```

最后一条命令应当没有输出。不要使用`git add -f`绕过`.gitignore`。

## 第一次提交

如果电脑尚未设置Git作者信息，先设置自己的真实姓名和GitHub邮箱：

```powershell
git config --global user.name "你的姓名或GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

如果不希望修改全局配置，可删除`--global`，只对当前仓库生效。

然后创建第一次提交：

```powershell
git add .
git commit -m "初始化LexChain图尔敏法律推理实验仓库"
```

## 在GitHub创建空仓库

1. 登录GitHub。
2. 点击右上角加号，选择“New repository”。
3. 建议仓库名使用`lexchain-toulmin-study`。
4. 根据数据管理要求选择Private或Public；首次上传建议先使用Private。
5. 不要在网页端勾选自动创建README、`.gitignore`或许可证，因为本地仓库已经包含这些文件。
6. 创建仓库后复制GitHub给出的HTTPS地址。

## 连接远程仓库并上传

把下面地址替换为自己的GitHub用户名和仓库地址：

```powershell
git remote add origin https://github.com/你的用户名/lexchain-toulmin-study.git
git branch -M main
git push -u origin main
```

GitHub通常要求使用浏览器登录、Git Credential Manager或Personal Access Token，不能把账户密码直接当作Git密码。凭据只保存在系统凭据管理器中，不要写入仓库文件。

## 后续更新

```powershell
git status
git add .
git commit -m "更新实验协议和数据集说明"
git push
```

## 发布前建议

- 初期将仓库设为Private。
- 确认Git历史中从未出现API Key或案件JSONL。
- 确认公开数据的许可、匿名化和伦理条件。
- 在论文投稿前创建不可变版本标签：

```powershell
git tag -a dataset-protocol-v0.1 -m "冻结数据集和实验协议v0.1"
git push origin dataset-protocol-v0.1
```

- 如果未来需要发布大文件，不要直接提交私有原始数据；优先发布下载脚本、校验摘要或经过授权的脱敏版本。

