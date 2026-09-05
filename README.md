# LexChain 图尔敏法律推理与评测

本仓库是论文实验的最小可复现Artifact，只保留固定数据集、推理方法、API运行工具、评测器及其协议。原始裁判文书、数据清洗流程和参考答案生产代码不在本仓库中。

## 数据集

`data/experiment_input_v1/`包含：

- `blind_development.jsonl`：300条开发集输入；
- `blind_test.jsonl`：2,861条测试集输入；
- `reference_all.jsonl`：3,161条QC参考答案，只能用于评分；
- `split_manifest.jsonl`：固定划分与分层；
- `freeze_report.json`：数据版本和摘要。

生成模型只能读取`blind_*`文件中的`model_input`，不得读取`reference_all.jsonl`。

## 推理脚本

- `scripts/main1_prompt_pilot.py`：Direct、CoT、IRAC、法律三段论、Schema、LexChain及单次强关系图尔敏。
- `scripts/toulmin_three_stage_batch.py`：三次调用、阶段锁定、无自动修复的图尔敏方法。
- `scripts/dashscope_batch_file.py`：提交、查询和下载Qwen Batch任务。

## 评测脚本

- `scripts/d7_api_scorer.py`：六维法律说理API评分，采用五个固定20分区间并允许档内整数分。
- `scripts/judgment_quantitative_scorer.py`：诉请、结果、金额、裁判操作和付款关系的100分确定性评分。

评分口径见`docs/D7_API_SCORING_PROTOCOL.md`与`docs/JUDGMENT_QUANTITATIVE_SCORING.md`，数据使用边界见`docs/DATASET_PROTOCOL.md`。

## 本地测试

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

测试不调用外部API。
