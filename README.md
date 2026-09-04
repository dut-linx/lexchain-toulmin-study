# LexChain 图尔敏法律推理实验

本仓库记录中国民事侵权案件中图尔敏启发法律推理研究的数据集协议、实验方案、评价设计和可复现处理代码。

## 当前进展

请求级参考标签生成阶段已经处理5,134条符合条件的案件。每条案件只进行一次Qwen Batch语义生成，5,134条全部成功返回，未出现API失败或JSON摄取错误。

经过确定性质量控制：

- 3,161条进入`qc_candidate`候选集；
- 1,973条进入`review_required`人工复核集；
- 这些结果仍属于机器辅助生成的参考标签候选，尚不能整体称为人工金标准。

结果盲的模型比较实验尚未开始。正式实验前必须完成数据版本冻结、信息泄漏检查、人工标注一致性校准、分组去重与数据切分，并预先锁定提示词和统计方案。

## 仓库结构

- `docs/EXPERIMENT_CHECKLIST.md`：分阶段实验清单、当前状态与验收标准。
- `docs/D7_API_SCORING_PROTOCOL.md`：依据六维标注手册改造的百分制API评分规则。
- `docs/DATASET_PROTOCOL.md`：数据集角色、构造方法、信息边界、QC和发布层级。
- `docs/PAPER_DATASET_SECTION_DRAFT.md`：可继续修改的论文数据集章节草稿。
- `docs/PAPER_EXPERIMENT_ALIGNMENT.md`：论文引言与实验清单之间的对应关系和设计缺口。
- `docs/GITHUB_UPLOAD.md`：初始化提交和上传GitHub的具体步骤。
- `manifests/dataset_v0.1.json`：当前私有数据文件的数量、大小和SHA-256摘要。
- `scripts/`：参考标签抽取、Batch摄取、质量控制和输入输出对齐脚本。
- `tests/`：不访问网络的确定性测试。

## 数据与隐私边界

案件原文、模型原始输出和API密钥不会进入Git仓库。`.gitignore`已经排除JSONL、Word、PDF、压缩包、环境变量文件及生成结果。仓库只公开代码、协议、汇总统计和文件摘要。

在公开任何案件文本前，必须确认数据来源许可、个人信息处理、匿名化要求和研究伦理或机构审批条件。

## 立即执行的里程碑

在运行MAIN-1之前，依次完成DATA-2至DATA-5和CAL-0。尤其不能把`court_reasoning`、`judgment_result`或根据原判选择的法条候选暴露给结果盲实验条件。

## 本地测试

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

生产数据测试在私有数据不存在时自动跳过，其余测试不需要API密钥，也不会调用外部服务。
