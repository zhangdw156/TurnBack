# TurnBack

[English](README.md) | 简体中文

本仓库是论文 **TurnBack: A Geospatial Route Cognition Benchmark for Large Language Models through Reverse Route** 的公开发布版本。

仓库只保留四类真正对外可用的内容：

1. `36kroutes/`：公开发布的原始路线语料
2. `path-builder execute`：公开版 Path Builder 执行器
3. `path-builder generate-routes`：`easy / medium / hard` 三档路线生成器
4. `path-builder generate-reverse`：面向外部大模型 API 的反转指令生成器

## 论文简介

TurnBack 把 **路径反转** 当作检验大模型地理空间认知能力的具体任务。模型首先拿到正向导航指令，目标是生成一条回到起点的反向路线指令。随后，我们用 Path Builder 把模型输出的反向指令重新执行成几何路线，再把恢复出的路线与参考反向路线做对比。

论文主要贡献有三部分：

- 提出了论文中定义的 `12` 个大都市、`36,000` 条步行路线的大规模路径反转 benchmark
- 提出了 Path Builder，把自然语言导航重新执行为街道级几何路径
- 提出了基于恢复几何而不是文本表面重叠的 route-level 评测方法

![TurnBack 流程图](assets/route_generation.png)


## 轻量 vLLM 评测工作流

本分支按相邻 K2、USTBench、STARK_Benchmark 的服务器评测方式整理。
在评测服务器上 fresh clone 后执行：

```bash
uv sync
bash scripts/prepare_eval_data.sh
```

`prepare_eval_data.sh` 会从 Hugging Face 下载 `zhangdw/TurnBack-10pct`，
把可断点复用的下载缓存放在 `${TMPDIR:-/tmp}` 下，并把评测 JSONL 安装到：

```text
data/turnback_10pct.jsonl
```

然后对已经用 vLLM/OpenAI-compatible chat completions 部署好的 Qwen3 模型评测：

```bash
bash scripts/run_vllm_eval.sh \
  --model qwen3-4b-thinking-2507 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --jobs 8
```

`--jobs` 是并发度开关。结果默认写到：

```text
results/<served-model-name>/turnback_10pct.jsonl
results/<served-model-name>/turnback_10pct_summary.json
```

评测脚本默认支持断点续评。重复运行同一命令时，只会跳过已经有完整结果的样本；
“完整”要求 sample id 匹配、数据源签名匹配、`parser_status=ok`，且 similarity 是有限数值。
中断造成的坏 JSONL 行、API 错误、解析失败、执行失败都不会被算作完成。
只有明确想全量重跑时才使用 `--force`。

用同一套完成判定预估进度：

```bash
uv run python scripts/estimate_eval_progress.py --model qwen3-4b-thinking-2507
```

实验完成后上传到 Hugging Face Buckets，并在另一台机器同步回来分析：

```bash
hfsync/local_to_remote.sh --dry-run
hfsync/local_to_remote.sh

# 分析机器上：
hfsync/remote_to_local.sh
```

默认目标是 `hf://buckets/zhangdw/leo-benchmark/TurnBack-eval/results`。
可用 `HF_BUCKET_ID`、`HF_TURNBACK_PREFIX`，或脚本的 `--bucket` / `--prefix` 覆盖。

## 快速开始

本地开发安装：

```bash
uv sync --extra dev --extra llm
```

生成三档新路线：

```bash
path-builder generate-routes \
  --city Toronto_Canada \
  --easy <NUM_EASY> \
  --medium <NUM_MEDIUM> \
  --hard <NUM_HARD> \
  --output-root tmp/generated_routes \
  --ors-api-key "$ORS_API_KEY"
```

用你自己的大模型 API key 生成反转指令：

```bash
path-builder generate-reverse \
  --provider openai \
  --city "Toronto, Canada" \
  --input-file 36kroutes/Toronto_Canada/easy/<ROUTE_ID>/natural_instructions.txt \
  --raw-output tmp/reverse_raw.txt \
  --clean-output tmp/reverse_clean.txt
```

用 Path Builder 执行反转指令：

```bash
path-builder execute \
  --root 36kroutes \
  --city Toronto_Canada \
  --difficulty easy \
  <ROUTE_ID> \
  --instructions tmp/reverse_clean.txt \
  --executor hybrid \
  --output tmp/recovered_route.geojson
```

将恢复路线与参考路线做相似度评分：

```bash
path-builder score \
  tmp/recovered_route.geojson \
  36kroutes/Toronto_Canada/easy/<ROUTE_ID>/route.geojson \
  --config configs/similarity.paper.json
```

## 主要结果

![论文主结果图](assets/main_results.png)

- 论文提出了一个覆盖 `12` 个大都市、总计 `36,000` 条路线、包含三档难度的 benchmark。
- 论文中 Path Builder 在 Toronto、Tokyo、Munich 上的成功率分别为 `96%`、`90%`、`94%`。
- 在代表性的 easy 反转样例上，没有模型能精确回到起点；Gemini 的相似度达到 `73.4`，Llama 为 `22.6`。
- 在 Toronto 的 `200` 条 easy 路线上，给 GPT-4o 增加向量地图提示后，return rate 从 `6.4%` 提升到 `43.7%`，similarity 从 `41.06` 提升到 `73.08`。

## 为什么目录还叫 `36kroutes`？

`36kroutes` 是论文时期沿用下来的发布名称。当前仓库中的这个目录不是重新命名后的“精确 36k 子集”，而是后续保留下来的原始发布快照。

论文中写的 `36,000` 条路线，对应的是论文中的 benchmark 定义。仓库保留 `36kroutes` 这个历史目录名，是为了保持已发布数据与代码的连续性。目录层面的说明见 [36kroutes/README.md](36kroutes/README.md)。

## 仓库结构

```text
.
├── 36kroutes/                # 已发布原始路线语料
├── assets/                   # 复用自论文的图
├── configs/similarity.paper.json
├── src/path_builder/         # Path Builder、路线生成、prompting、评分
├── scripts/                  # 数据准备、vLLM 评测、进度预估、本地 smoke test
├── hfsync/                   # HF bucket 上传/下载脚本
├── tests/                    # 公开测试集
├── README.md                 # 英文 README
├── README.zh-CN.md           # 中文 README
├── CITATION.cff
└── pyproject.toml
```

## 快速检查

```bash
uv run ruff check src/path_builder tests scripts --select F,E9
uv run python -m compileall src/path_builder scripts
uv run pytest -q
bash scripts/quick_check.sh
```

## 引用

如果你使用了本仓库的代码或数据，请引用 TurnBack 论文。引用元信息见 [CITATION.cff](CITATION.cff)。
