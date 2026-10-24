# CXR First 十类胸片竞赛管线

本文件记录 `codex/cxr-first-10class` 分支对竞赛数据集与《比赛参赛规则》的实现边界。
数据集与规则原文由赛事主办方提供，仓库不复制患者影像、测试标签或主办方材料。

## 一、数据概况（已在本机核实）

数据根目录（默认）：`D:\医疗影像识别\cxr_first\cxr_first`

| 划分 | 图像数 | 检查数（Study） | 患者数（Subject） | 标签 |
| --- | ---: | ---: | ---: | --- |
| train | 30,344 | 18,513 | 11,055 | 多标签（`Predict_class`，`;` 分隔） |
| valid | 1,422 | 859 | 228 | 多标签 |
| test | 2,218 | 1,414 | 136 | 无标签 |

- 目录结构：`<split>/<Subject_id>/<Study_id>/<Dicom_id>.jpg`。
- 三个划分的患者集合两两无交集（`stats` 命令会校验，交集非零直接报错）。
- 一次检查可能含 1–7 张图像（train 平均 1.64 张）；`ViewPosition` 含 AP/PA/LATERAL/LL 与缺失值。
- 图像为可变尺寸 JPG（约 0.7–3.5 MB），竞赛评测为十类多标签，主指标宏平均 AUC、辅助指标宏平均 F1。
- 标签为 1–10 的数字 id，`class 10` 在 train 中只单独出现（12,649 张，41.7%），
  初步判断对应“无异常/正常”，最终以主办方类别映射表为准（见“待确认”）。

## 二、竞赛规则与本实现的对应

| 规则 | 要求 | 本分支实现 |
| --- | --- | --- |
| 第 1 条 预训练模型 | 只允许 ImageNet/COCO/LAION 等非医学开源权重；不得使用医学预训练；需提供权重链接、许可证全文、上游数据说明、SHA-256 | 仅使用 torchvision 官方 ImageNet DenseNet-121（`pretrain_model/` 保存权重与许可证，`model.py` 记录来源）；训练脚本以 `--no-pretrained` 支持离线冒烟 |
| 第 2 条 训练数据 | 只允许主办方训练集做有监督微调，禁止外部医学数据 | `train` 只读取 `train.csv` + `train/`；不引用 CheXpert/MIMIC 代码 |
| 第 3 条 验证集 | 可用于早停/checkpoint/阈值/超参；并入训练后不得再用于选择 | `valid` 仅用于早停、最佳 checkpoint 与 F1 阈值固定 |
| 第 4 条 数据增强 | 允许几何/像素/遮挡/Mixup/重采样/类别权重，仅训练期使用 | `data.build_transforms` 仅在 `training=True` 启用随机裁剪/旋转/亮度对比度；推理为确定性 Resize+CenterCrop |
| 第 5 条 测试集 | 禁止任何形式使用测试集训练/统计/选模 | `predict --split test` 只做前向推理；`stats`/`train`/`evaluate` 不读取测试标签 |
| 第 6 条 推理约束 | 固定参数、eval 模式、单图确定性预处理、阈值固定、TTA/集成可选 | 推理 `model.eval()` + `torch.no_grad()`；每图独立归一化；阈值来自 valid；默认关闭 TTA |
| 第 7 条 耗时 | 单图端到端 ≤100ms，100 图 batch=1 × 3 轮，`cost.py` 与 `predict.py` 共用推理逻辑 | 后续在 `predict` 之上实现 `cost.py`（与提交材料一起打包），本分支先生成 submission |
| 第 8 条 模型大小 | 解压后权重+推理配置 ≤500MB | DenseNet-121（约 30MB）满足 |
| 第 9/10/11 条 | 提交/撤销、可复现、违规认定 | 训练结果含 commit、配置、阈值、checkpoint SHA-256，可复现 |
| 附录 A | 提交包：`code/`、`model/`、`pretrain_model/`、`licenses/`、`cost.py`、`模型和推理相关材料清单.xlsx`、`submission.csv` | 打包脚本与 xlsx 清单待补（见“下一步”） |

## 三、命令

```powershell
# 环境（首次）
.\.venv\python.exe -m pip install -e ".[dev,ml]"

# 数据集统计（会校验跨划分患者泄漏）
.\.venv\python.exe -m cxr_competition stats --data-root cxr_first\cxr_first

# 本地 CPU 冒烟（小样本，验证链路；不得作为正式结果）
.\.venv\python.exe -m cxr_competition train --data-root cxr_first\cxr_first `
  --output-dir runs\cxr-first-smoke --device cpu --no-pretrained `
  --image-size 128 --epochs 1 --batch-size 4 --num-workers 0 `
  --max-train-images 8 --max-valid-images 4

# 正式训练（租用 GPU 后；使用 ImageNet 权重）
.\.venv\python.exe -m cxr_competition train --data-root cxr_first\cxr_first `
  --output-dir runs\cxr-first-densenet121 --device cuda --pretrained `
  --image-size 384 --epochs 8 --batch-size 32

# 验证集评估（宏平均 AUC / 宏平均 F1）
.\.venv\python.exe -m cxr_competition evaluate --checkpoint runs\cxr-first-densenet121\best.pt

# 测试集推理（按图像或按检查聚合）
.\.venv\python.exe -m cxr_competition predict --checkpoint runs\cxr-first-densenet121\best.pt `
  --split test --output submissions\submission.csv --device cuda
.\.venv\python.exe -m cxr_competition verify-submission --submission submissions\submission.csv
```

## 四、下一步（按依赖顺序）

1. **确认官方映射**：10 个类别 id→中英文名（写入 `ml/cxr_competition/labels.py` 的 mapping 文件）以及
   `submission.csv` 的准确列格式（按 `Dicom_id` 还是按 `Study_id`、列名、精度）。
2. **补提交材料**：`cost.py`（100 图 × 3 轮、batch=1、与 `predict` 共用推理）、
   `pretrain_model/` 权重与 SHA-256、`licenses/` 许可证全文、`模型和推理相关材料清单.xlsx`、
   打包脚本与目录结构。
3. **正式训练**：租 GPU 后跑 DenseNet-121（ImageNet 初始化），验证集选模；
   记录实际耗时、显存、精度（FP32/FP16）与费用。
4. **平台集成（可选）**：若要把十类模型接入桌面端，需要把 `contracts/labels.json` 迁移为十类契约，
   并同步推理服务与 UI；该改动会触碰主分支契约，需单独评审。

## 五、待确认（阻塞项）

- 10 个类别的 id→名称映射表（用户消息中的 `image.png` 未随文本提供）。
- 测试提交 `submission.csv` 的列定义与评价粒度（图像级 vs 检查级）。
- 正式训练 GPU 预算与租用方式；本机无 CUDA（torch 2.14.0+cpu）。
- 赛事平台与截止时间（区域赛/省赛），以及是否需要 `cost.py` 先于 submission 提交。
