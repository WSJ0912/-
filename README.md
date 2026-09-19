# 医疗影像智能识别平台

成人正位 AP/PA 胸部 X 光片多标签筛查的科研与竞赛原型。项目同时提供可复现的 CheXpert 研究管线、本地 FastAPI/ONNX Runtime 推理服务，以及 Electron + React + Cornerstone3D Windows 阅片工作区。

> **重要限制**：本项目不是医疗器械，也不是独立诊断工具。AI 输出必须由经过培训的专业人员复核。仓库不包含患者影像、受控数据、真实模型权重或虚构性能指标。

## 已实现范围

- 固定 CheXpert 14 项标签、患者级划分、不确定标签掩码和带上限类别权重的 masked BCE。
- ImageNet DenseNet-121 基线与唯一改进 MixStyle（`p=0.5`、`alpha=0.1`），固定 3 个种子、最多 15 轮和六次正式运行协议。
- MIMIC 固定种子抽样、20 GiB 总空间预算、SHA-256 封存和聚合-only 外部评价。
- 每类 AUROC、AUPRC、敏感度、特异度、F1、Brier，以及患者级 bootstrap 95% 置信区间。
- 强制验证集最优 MixStyle 部署选择、ONNX logits/特征图导出及 `1e-4` 概率一致性门禁。
- `.medmodel` / `.medexperiment` 清单、文件哈希、固定标签和 ZIP 路径安全校验。
- 本地随机端口、双令牌、Argon2id、角色权限、SQLite、DICOM 去标识化、烧录文字遮挡和 EXIF 清理。
- 检查队列、Cornerstone3D 阅片、14 项医生复核、报告版本/PDF、实验结果、模型管理和管理员设置。
- 可选 OpenAI Responses API 文字助手，固定 `store:false`、Structured Outputs、无工具调用；离线核心流程完整可用。
- PyInstaller 本地服务和 electron-builder Windows x64 NSIS 构建链。

## 当前实证状态

自动化测试使用无患者信息的合成 DICOM、测试模型和真实 ONNX Runtime。以下事项尚未执行，不能由代码或合成测试替代：真实 CheXpert 训练、六次正式 GPU 实验、MIMIC 外部评价、300 元 GPU 预算核验、Ryzen 5 5600 性能测试、医生约 50 项流程评价、干净 Windows 10/11 安装/卸载，以及最终权重分发许可确认。

发布状态见 [v0.1 发布门禁](docs/RELEASE_GATES.md)。

## 目录

```text
contracts/                 共享标签、类型与 JSON Schema
ml/cxr_research/           数据、模型、训练、指标、外部评价和导出
services/inference/        本地推理、匿名化、存储、权限和报告服务
apps/desktop/              Electron/React/TypeScript/Cornerstone3D 桌面端
tests/python/              合成数据单元与端到端 API 测试
docs/                      架构、治理、研究、安全、构建和发布门禁
third_party/               参考项目与锁定依赖许可清单
```

## 快速开始

需要 Python 3.11 和 Node.js 20/22：

```powershell
cd D:\医疗影像识别
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml,service,build]"
cd apps\desktop
npm ci
npm run dev
```

没有许可明确且训练完成的 `.medmodel` 时，软件会显示“尚未安装模型”并拒绝预测，不会生成随机或演示结果。

## 验证与构建

```powershell
cd D:\医疗影像识别
.\.venv\Scripts\python.exe -m ruff check ml services tests\python
.\.venv\Scripts\python.exe -m unittest discover -s tests\python -v
cd apps\desktop
npm run build
npm audit --audit-level=low
npm run dist
```

`npm run dist` 生成 Windows x64 NSIS 安装包；目标电脑不需要 Python、Node 或 NVIDIA 驱动。详细步骤和未完成的目标机验证见 [Windows 构建说明](docs/WINDOWS_BUILD.md)。

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [数据治理](docs/DATA_GOVERNANCE.md)
- [研究协议](docs/RESEARCH_PROTOCOL.md)
- [安全设计](docs/SECURITY.md)
- [Windows 构建](docs/WINDOWS_BUILD.md)
- [10 周路线](docs/ROADMAP.md)
- [发布门禁](docs/RELEASE_GATES.md)
- [第三方声明](third_party/NOTICE.md)

## 许可

本仓库源码采用 Apache-2.0，见 [LICENSE](LICENSE)。CheXpert、MIMIC-CXR-JPG、ImageNet 初始化权重、最终模型权重和第三方运行库分别受其原始条款约束。
