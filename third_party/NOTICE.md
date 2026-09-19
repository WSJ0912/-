# 第三方参考与声明

本项目只参考公开项目的接口设计、论文方法和公开文档，不复制许可不明的代码或权重。以下信息已于 2026-09-19 从对应 GitHub 仓库的 LICENSE/README 复核；实际锁定依赖另见 `DEPENDENCY_LICENSES.md`。

| 项目 | 许可证与权利人 | 本项目用途与边界 |
| --- | --- | --- |
| [Cornerstone3D](https://github.com/cornerstonejs/cornerstone3D) | MIT；Copyright (c) 2019 Open Health Imaging Foundation | 桌面端直接依赖 `@cornerstonejs/*`，用于本地影像显示。安装包保留依赖许可。 |
| [TorchXRayVision](https://github.com/mlmed/torchxrayvision) | 仓库声明主库 `xrv.models`、datasets、utils 等为 Apache-2.0；`xrv.baseline_models` 要逐模型检查许可 | 只参考胸片 API、预处理和标签设计，不作为运行依赖，不复制 baseline 权重。README 明确多数预训练 DenseNet 有 18 个输出，部分数据集中未训练的输出会随机预测，因此绝不能重命名为本项目固定 CheXpert 14 输出。 |
| [MixStyle](https://github.com/KaiyangZhou/mixstyle-release) | MIT；Copyright (c) 2021 Kaiyang Zhou | 依据公开论文和方法说明重新实现最小 MixStyle 模块，固定 `p=0.5`、`alpha=0.1`；未复制训练工程或权重。 |

## 数据和权重

- CheXpert 数据集的访问和使用受 Stanford 发布条款约束，不随本仓库分发。
- MIMIC-CXR-JPG 受 PhysioNet 凭证访问和数据使用协议约束，不随本仓库分发。
- ImageNet 初始化权重、训练得到的模型权重和任何第三方 baseline 权重必须逐项确认再分发权；源码 Apache-2.0 不自动覆盖它们。

## 二进制依赖

`scripts/generate-dependency-inventory.py` 根据当前 Python 环境和 `package-lock.json` 生成 `DEPENDENCY_LICENSES.md`。每个候选安装包都必须重新生成并人工处理“未解析”项。electron-builder 将项目许可证和 `third_party` 文档复制到 `resources/legal`。
