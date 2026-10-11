# 使用说明

在 `config.json` 设置模型、数据、服务和训练参数；API 密钥通过配置指定的环境变量提供。

```bash
R2AI_SETUP_PYTHON=python3.12 bash scripts/setup.sh full
bash scripts/run.sh --config config.json --check
bash scripts/run.sh --config config.json
bash scripts/report.sh --run runs/research
bash scripts/evaluate_full.sh --config config.json --run runs/research
```

`--check` 不调用 API 或启动 GPU。搜索与完整评测分别运行。

| 内容 | 文档 |
|---|---|
| 轮数、预算、门控、S1–S4 | [参数配置](research_configuration.md) |
| 安全数据、攻击生成、通用数据 | [数据准备](data_interface.md) |
| 训练后端、adapter 导出 | [模型后端](model_backends.md) |
| benchmark 数据与评测设置 | [完整评测](formal_evaluation.md) |
| 已验证范围 | [验收记录](acceptance.md) |

搜索结果见 `runtime.output` 下的 `summary.json` 和生成的 `report/trials.csv`；评测结果见 `final_evaluation.output` 下的 `summary.json`、`metrics.csv` 及逐条记录。相同命令可续跑，修改输入或执行代码后需更换输出目录。
