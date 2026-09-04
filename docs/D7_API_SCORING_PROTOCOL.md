# D7百分制API评分方案

## 评价对象

本方案评价模型针对侵权纠纷一审案件生成的请求级结论与法律说理。每次评分向评审API提供结果盲案件材料、冻结的QC参考答案和一份待评模型输出。法院原始裁判说理和裁判结果不作为被测模型输入，也不作为评审API的额外答案来源。

## 六个维度

评分沿用《侵权纠纷一审 D7 六个主维度标注手册》的判断内容：

1. D1 诉请、主体与决定性争点覆盖。
2. D2 责任规范选择、法源适格与解释。
3. D3 事实、规范与法律效果连接。
4. D4 抗辩、例外与责任限缩处理。
5. D5 请求级结论、责任分配与救济证成。
6. D6 整体公共证立与可审查性。

每个维度改为0至100整数。原手册四级锚点对应0、33、67和100分，API可以在相邻锚点之间取整数。D4在案件材料确无实质抗辩时记为N/A；模型遗漏已经出现的抗辩不能记为N/A。

## 汇总分数

- `substantive_score`：D1至D5适用维度的算术平均数，是主要人工语义评价指标。
- `overall_score`：D1至D6适用维度的算术平均数，用于完整质量报告。
- 六个维度必须分别报告，汇总分不能替代分维度结果。

不人为设置维度权重。D4为N/A时从分母中排除。这样既保留手册中D6的辅助性质，也避免主观权重掩盖具体法律错误。

## 严重错误上限

- D2出现虚构、失效或明显不适格法源时，D2最高25分。
- D3依赖材料外核心事实时，D3最高25分。
- D5与QC参考答案在主要诉请结论、责任主体、责任形态或金额上实质冲突时，D5最高25分。

## API与本地程序的职责

API负责理解案件、适用六维手册、给出百分制分数、理由和双方证据。每个案件和每个待比较条件独立评分。正式运行使用Qwen Batch Chat，温度为0并关闭思考输出。

本地程序不改写API的语义判断，只执行JSON校验、0至100范围校验、D4的N/A校验、总分重算、错误隔离和Token统计。任何解析失败或字段缺失均进入错误文件，不以0分代替。

## 运行接口

先把某一种实验条件的输出整理为JSONL。每条至少包含`case_id`，并在`model_output`、`candidate_output`、`prediction`或`output`之一保存完整待评输出。然后生成Qwen Batch Chat请求：

```powershell
python scripts/d7_api_scorer.py prepare `
  --blind C:\Users\Lin\Desktop\fx\AAAI\experiment\qc3161_v1\blind_development.jsonl `
  --references C:\Users\Lin\Desktop\fx\AAAI\experiment\qc3161_v1\reference_all.jsonl `
  --candidates <某一实验条件的输出.jsonl> `
  --output <D7评分Batch输入.jsonl> `
  --model qwen3.7-max
```

Batch完成后进行格式校验和总分重算：

```powershell
python scripts/d7_api_scorer.py ingest `
  --batch-result <D7评分Batch返回.jsonl> `
  --output <D7百分制评分.jsonl> `
  --errors <D7评分错误.jsonl> `
  --report <D7评分汇总.json>
```

开发集可以反复调试。正式测试集生成模型输出和API评分后，不再根据测试结果修改提示词或评分规则。

## 可靠性控制

正式实验前从开发集选取共同样本进行重复评分，检查同一输出的分数稳定性。正式报告至少包括API模型名称、运行日期、提示词版本、参数、每维均值、置信区间、评分失败率和Token费用。建议另抽取一部分案件由法律评审者独立评分，用于估计API评分与人工评分的一致性。
