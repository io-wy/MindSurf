# MindSurf 项目约束指南

本文件是 MindSurf 任务的统一执行入口。新线程开始时必须完整阅读；任何暂存、
提交、推送、Pull Request、发布或服务器变更前必须再次核对相关章节。
这些约束只适用于本仓库，不写入或修改机器全局代理配置。

## 1. 约束优先级

发生冲突时按以下顺序处理：

1. 用户在当前任务中明确指定的范围和目标；
2. GitHub 上目标分支的实时状态；
3. 本文件；
4. 仓库中的专项文档；
5. 旧交接记录、旧报告和本地历史状态。

旧记录只能提供线索，不能替代实时核验。涉及分支、服务器、GPU、数据、
checkpoint 或发布状态时必须重新检查。

## 2. 新线程启动检查

开始任何修改前：

1. 完整阅读本文件；
2. 阅读根目录 `AGENTS.md` 和任务涉及的 README、配置或专项文档；
3. 明确本次目标、非目标、允许修改的目录和最终交付位置；
4. 检查 Git 工作区、分支、remote 和远端基线；
5. 检查未跟踪文件，默认将其视为用户资产；
6. 对会变化的外部状态执行实时核验。

建议命令：

```powershell
git status -sb
git branch -vv
git remote -v
git fetch --all --prune
git rev-parse HEAD
git ls-remote https://github.com/io-wy/MindSurf.git refs/heads/main refs/heads/pretrain
```

remote 名称和本地分支可能过期，判断目标时以 URL、远端 SHA 和用户指定分支为准，
不能只看 `origin`、`upstream` 等本地别名。

## 3. 分支与代码树边界

- `main` 是稳定分支。除非任务明确属于 release 或 hotfix，并得到明确授权，否则
  不修改、不合并、不推送 `main`。
- `pretrain` 是预训练工作的当前集成基线。涉及该方向时，以
  `io-wy/MindSurf:pretrain` 的实时文件树为准。
- 本地旧 MiniMind worktree、报告和 checkpoint 只作为证据来源；与远端树冲突时，
  不得用旧树覆盖或替换 `pretrain`。
- 禁止通过历史嫁接、整仓复制或大规模 vendor 导入把另一项目的文件树伪装成增量
  功能。

小型文档和低风险修正，在用户明确要求直接落到 `pretrain` 时，可以从最新
`pretrain` 创建干净工作区，完成单一提交后直接推送。功能开发、重构、数据管线、
训练逻辑和其他高风险改动默认使用短期 `feature/<kebab-case-name>` 分支并通过
Pull Request 合入 `pretrain`。

不得在没有说明用途和收尾方式时创建远程临时分支。临时分支合并后应立即删除；
删除前确认其有效提交已包含在目标分支：

```powershell
git merge-base --is-ancestor <temporary-branch> <target-branch>
```

## 4. 工作区纪律

- 保留用户已有的修改和未跟踪文件；
- 不使用 `git reset --hard`、无目标的清理命令或强制覆盖；
- 不为追求“干净”而删除未知数据、日志、环境文件或 checkpoint；
- 搜索和修改只围绕当前任务，不顺手重构无关模块；
- 风险较高或文件树差异较大的工作使用独立 worktree；
- 删除或移动 worktree 前先解析绝对路径，确认目标位于预期工作目录且状态干净。

暂存必须逐文件执行：

```powershell
git add path/to/file1 path/to/file2
```

禁止使用 `git add -A` 或未经核对的全量暂存。

## 5. 代码和实验原则

- 从数据身份开始：冻结 manifest、split、tokenizer、seed 和评测口径；
- 对照实验一次只改变一个主要变量，并固定 seen tokens 或明确报告预算差异；
- 同时记录 loss、吞吐、峰值显存、参数量、数据哈希和配置身份；
- 结论必须区分“最低 loss”“实际有效差异”和“能力门通过”；
- checkpoint、数据和评测证据缺失时，状态写为缺失，不从报告推断其存在；
- 不得在未经明确要求时启动训练或覆盖已有 run；
- 新数据必须使用新的 manifest 和 run identity，不与旧数据结果混表。

## 6. README 和公开文档标准

README 面向项目参与者和使用者，应优先回答：

1. 这个分支解决什么问题；
2. 已经完成了哪些工作；
3. 当前结果和关键指标是什么；
4. 哪些能力尚未达到；
5. 如何找到复现配置、报告和公开产物；
6. 下一阶段的技术目标是什么。

README 不承载代理工作过程、用户指令、内部协作对话、提交纪律复述或个人环境说明。
工程约束放在本文件或贡献指南中；实验细节放在报告和机器可读结果中。

公开文档应直接陈述事实，不使用以下写法：

- “我检查了”“用户要求”“下一步我会”等对话式过程；
- “旧树没有复制”“为避免之前错误”等内部事故叙述；
- 无法在公开仓库验证的本地 commit、绝对路径或环境快照；
- 没有负责人、验收标准或依据的 TODO、TBD、placeholder；
- 为展示工作量而增加的模板章节、重复结论和冗长教学说明。

## 7. 团队交付内容清理

可以提交：

- 经审查的源码、配置、迁移、自动化脚本和测试；
- 面向维护者或使用者的 README、接口、部署和故障处理文档；
- 可复现实验所需的参数、manifest、阈值、指标摘要和数据来源；
- 小型测试 fixture、CI 配置和依赖声明。

不得提交：

- Codex、Claude、ChatGPT、Superpowers 等工具的 prompt、计划和执行记录；
- `.codex/`、`.agents/`、`.codebase-memory/`、`.superpowers/` 等本地状态；
- task brief、review package、进度 ledger 或仅供代理续接的临时报告；
- 个人用户名、内网地址、绝对路径、SSH 细节、凭据和环境快照；
- 数据集、checkpoint、虚拟环境、缓存、原始 profiler 数据和瞬时日志；
- 与任务无关的格式化、重命名、目录迁移或占位实现。

每个新增文件必须能够说明维护者、用途和删除后的实际损失。

## 8. 大文件与实验资产

普通 Git 只保存代码、配置、小型 fixture、manifest 和汇总结果。以下内容使用
DVC、Release、对象存储或项目指定的外部资产系统：

- 完整训练数据；
- `*.pth`、`*.pt`、`*.ckpt`、`*.safetensors`；
- 训练 run 目录、原始日志、PID、tmux 状态和 profiler 数据库；
- 虚拟环境、wheelhouse 和依赖缓存。

缺失资产必须显式标记为 `missing`。存在报告不代表对应权重或数据仍可获取。

## 9. 服务器操作

服务器连接信息保留在本地交接文档，不写入公开仓库。服务器任务遵循：

- 只在授权目录内工作；
- 操作前核对用户、主机、工作目录、GPU 和当前进程；
- GPU 忙碌时不抢占，不影响其他用户进程；
- 未经明确要求不启动训练；
- 同步源码前备份将被覆盖的文件；
- 不删除 dataset、checkpoint、out、logs 或未知实验产物；
- 训练中断后先核对 tmux、进程、GPU、日志和输出，再决定是否续跑；
- 服务器结论以实时状态为准。

## 10. 提交前差异审计

先确定精确基线，再审查总体差异：

```powershell
git diff --stat <base>...HEAD
git diff --name-status <base>...HEAD
git diff --summary <base>...HEAD
git log --oneline <base>..HEAD
```

出现以下任一情况必须停止：

- 与任务无关的顶层目录被删除或重命名；
- API、Infra、CI、测试或迁移等核心模块批量消失；
- 小型任务修改数百文件或出现异常新增/删除行数；
- diff 呈现整仓替换、历史嫁接或无边界 vendor 导入；
- 无法逐项解释删除文件；
- 验证没有覆盖受影响模块。

## 11. 暂存后发布门禁

显式暂存后执行：

```powershell
git status -sb
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```

检查过程文件和个人环境信息：

```powershell
git diff --cached --name-only |
  Select-String -Pattern 'superpowers|\.codex|\.agents|\.superpowers|task-brief|review-package' -CaseSensitive:$false

$stagedFiles = git diff --cached --diff-filter=ACMR --name-only -- . `
  ':(exclude)PROJECT_RULES.md' ':(exclude)AGENTS.md'

$stagedFiles | ForEach-Object {
  Select-String -LiteralPath $_ `
    -Pattern 'Codex|Claude|ChatGPT|/home/[A-Za-z0-9_-]+|192\.168\.|C:\\|D:\\|DPAPI' `
    -CaseSensitive:$false
}
```

检查凭据和模型文件：

```powershell
git diff --cached --name-only |
  Select-String -Pattern '\.(pth|pt|ckpt|safetensors|env|key|pem)$' -CaseSensitive:$false

git grep --cached -n -I -E '(BEGIN (RSA|OPENSSH) PRIVATE KEY|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,})' -- .
```

命中必须逐项解释；属于过程、个人环境、凭据或意外大文件的内容先删除或泛化。

## 12. 验证、提交与推送

验证强度与风险匹配，至少包括：

- 修改文件的格式和语法检查；
- 受影响模块的 focused tests；
- 仓库规定的 lint、type check 和 pytest；
- 文档链接、JSON/YAML 解析和生成物幂等检查；
- `git diff --check`；
- 凭据、个人信息和大文件扫描。

若完整质量门因基线问题失败，应提供可复现证据并明确区分本次变更与既有问题，不得写成
“全部通过”。

提交使用 Conventional Commits。推送前再次确认：

1. 远端目标分支 SHA 未漂移；
2. 暂存范围只有当前任务；
3. commit 的父提交是预期基线；
4. 推送 refspec 指向正确分支；
5. `main` 不在本次 refspec 中。

推送后从远端反查 SHA、文件和目标分支。涉及 `pretrain` 的直接修正，还要比较操作前后的
`main` SHA，确认未变化。禁止对共享分支使用 force push。

## 13. Pull Request 与发布

功能 PR 应列出：

- 目标和明确非目标；
- 修改和删除的目录；
- `git diff --stat` 摘要；
- 测试与安全扫描证据；
- 数据或配置身份；
- 迁移和回滚方式；
- 已知限制。

永久分支的功能合并默认需要至少一名非作者审查者和状态检查。release 或 hotfix 合入
`main` 后必须同步回开发分支并打版本 Tag。合并完成后删除临时分支。

## 14. 任务结束与交接

最终报告只陈述可验证结果：

- 修改文件和 commit SHA；
- 推送或 PR 的准确目标；
- 测试、扫描和远端核验结果；
- 数据、checkpoint、服务器或 CI 的已知缺口；
- 是否修改了 `main`；
- 是否仍有临时分支、worktree 或后台任务。

用户要求暂停时，完成当前交付核验后停止，不继续开展下一阶段工作。

## 15. 最终核对表

提交或发布前逐项确认：

- [ ] 已重新阅读本文件；
- [ ] 目标分支、基线 SHA 和任务范围明确；
- [ ] `main` 不在修改或推送范围；
- [ ] 未跟踪的用户资产保持原状；
- [ ] diff 不包含整仓替换或无关删除；
- [ ] README 面向项目参与者，内容与实际成果一致；
- [ ] 无代理过程、个人路径、内网信息、凭据和意外大文件；
- [ ] 数据、配置、实验和模型状态均有证据；
- [ ] focused tests 与适用质量门已运行；
- [ ] 暂存差异和提交历史已检查；
- [ ] 推送后已反查远端 SHA 和目标文件；
- [ ] 临时分支、worktree 和后台任务已按约定收尾。
