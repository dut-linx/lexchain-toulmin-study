# 实验输入数据

`experiment_input_v1/`保存已经冻结的模型输入、划分信息与QC参考答案。模型输入本身不包含`gold_output`、原判结果或法院说理。

- `blind_development.jsonl`：提示词开发和先导实验。
- `blind_test.jsonl`：方法冻结后的最终检验。
- `split_manifest.jsonl`：固定划分与分层信息。
- `reference_all.jsonl`：3161条通过QC的参考答案，用于复现评分；不得作为生成模型输入。
- `freeze_report.json`：版本、数量与摘要信息。

开发集与测试集共同构成3161条输入，因此不重复提交`blind_all.jsonl`。原始裁判文书及参考答案生产脚本仅保存在本地。公开`reference_all.jsonl`后，测试答案不再隐藏；正式测试应在答案发布前完成，或另设不公开测试集。
