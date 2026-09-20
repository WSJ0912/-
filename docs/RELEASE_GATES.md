# v0.1 发布门禁

状态说明：`PASS` 表示已有可重复证据，`PENDING` 表示必须由真实数据、目标硬件或人工评价完成，`BLOCKED` 表示该项未完成前不得发布相应资产或结论。

## 自动化工程门禁

| 门禁 | 要求 | 当前状态 |
| --- | --- | --- |
| E-01 | Ruff、compileall、Python 单元/集成测试通过 | PASS（合成数据与真实 ONNX Runtime 路径） |
| E-02 | Electron main/preload/IPC policy 语法、TypeScript、Vite build 通过 | PASS（2026-09-19 本地候选构建） |
| E-03 | `npm audit --audit-level=low` 无已知漏洞 | PASS（2026-09-19，0 个已知漏洞） |
| E-04 | `.medmodel` 固定 14 输出、全部哈希、ZIP 路径安全 | PASS（自动化测试） |
| E-05 | PyTorch/ONNX 概率最大误差 `<=1e-4`，logits、特征图与 14 类 CAM 形状正确 | PASS（随机初始化真实 DenseNet 回归测试；不代表医学合理性验证） |
| E-06 | 导入、匿名化、真实 ONNX Runtime、复核、报告锁定、中文 PDF | PASS（合成工作流 API 测试） |
| E-07 | 新 NSIS 安装包构建成功并包含 `resources/legal` | PASS（2026-09-19，本地候选构建、开发机静默安装/卸载及内嵌服务工作流验证） |
| E-08 | Git diff 无空白错误，仓库不含数据、权重、密钥、数据库或构建物 | PASS（2026-09-19 提交前扫描；推送前再次检查） |

## 研究门禁

| 门禁 | 要求 | 当前状态 |
| --- | --- | --- |
| R-01 | CheXpert 许可和成人 AP/PA 患者级清单锁定 | BLOCKED：需要真实数据访问 |
| R-02 | 基线与 MixStyle 各 3 个固定种子，共 6 次正式运行 | BLOCKED：尚未运行 |
| R-03 | 记录 GPU 实际费用且总额不超过 300 元 | BLOCKED：尚未租用 GPU |
| R-04 | 只按 CheXpert 验证五项 macro AUROC 选择单个 MixStyle seed | 工具已强制；真实选择 PENDING |
| R-05 | 最终模型 ONNX 一致性、阈值、模型卡和权重许可确认 | BLOCKED：尚无最终权重 |
| R-06 | 5,000-8,000 名成人 MIMIC 清单先封存后评价 | BLOCKED：需要受控数据访问 |
| R-07 | 外部指标、全部患者级 95% CI 和失败案例复核 | BLOCKED：尚未评价 |

## 软件与人工门禁

| 门禁 | 要求 | 当前状态 |
| --- | --- | --- |
| W-01 | 干净 Windows 10 x64 无 Python/Node 安装、推理、卸载 | BLOCKED：未执行 |
| W-02 | 干净 Windows 11 x64 无 Python/Node 安装、推理、卸载 | BLOCKED：未执行 |
| W-03 | Ryzen 5 5600 最终模型单张 CPU 推理不超过 15 秒 | BLOCKED：未实测 |
| W-04 | 约 50 个合成或许可明确任务的医生流程评价 | BLOCKED：需要医生参与 |
| W-05 | 只评价术语和工作流，不把该评价表述为诊断准确率验证 | PENDING |
| W-06 | 安装包签名、卸载数据策略和第三方许可人工复核 | PENDING：候选安装包尚未使用 Authenticode 证书签名 |

## 发布决策

- 可以在工程门禁通过后推送源码，明确标记为科研原型并列出未完成门禁。
- 不得在 R-01 至 R-07 未完成时发布真实性能结论或医学模型权重。
- 不得在 W-01、W-02 和 W-06 未完成时把 NSIS 资产标记为正式稳定发行版。
- 当前 Actions 会让严格稳定 SemVer 标签（含 build metadata）校验失败；仓库还必须用 GitHub ruleset 要求该检查并限制 Release 创建，单独的 workflow 红灯不能阻止手工发布。
- `v0.1.0` Release 创建前，由维护者在本文件记录证据链接、日期和执行人；不能用合成测试替代真实数据、目标硬件或医生评价。

2026-09-19 本地候选安装包仅用于工程验证，未提交到 Git、未作为 Release 发布。其 SHA-256 为 `276E8578DFFA79711CBB6046EAE1F00543DA6247E3AF512C3302640C866E7972`；任何重建都会改变该哈希，发布时必须重新记录。
