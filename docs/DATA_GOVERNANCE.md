# 数据治理

## 数据范围

- 训练：经授权取得的 CheXpert 小图版，且只纳入成人正位 AP/PA 影像。
- 外部测试：经授权取得的 MIMIC-CXR-JPG，只用于一次封存的最终外部评价。
- 应用导入：成人胸部 X 光正位 AP/PA DICOM、PNG 或 JPEG。
- 仓库：只允许源码、配置、合成测试、聚合指标和模型卡，不允许患者影像、受控标识、患者级预测或真实权重。

CheXpert 和 MIMIC-CXR-JPG 的取得、存储和使用必须遵守各自条款。Apache-2.0 只覆盖本仓库源码，不授予数据或模型权重分发权。

## 20 GiB 空间预算

`prepare-chexpert` 会在患者级划分清单中记录 CheXpert 影像总字节数。`sample-mimic` 必须接收该清单，并从 20 GiB 总预算中计算 MIMIC 剩余额度：

```powershell
python -m cxr_research sample-mimic metadata.csv data\mimic-sealed.json `
  --image-root D:\datasets\mimic-cxr-jpg `
  --chexpert-manifest data\chexpert-split.json `
  --target-patients 8000
```

抽样先用固定种子打乱成人患者，每人选择一张 AP/PA 影像，再仅按固定顺序和文件大小收缩列表。过程不得读取结局标签或模型表现。若预算内不足约 5,000 人，必须记录偏差，不能查看结果后重新抽样。

## 封存与泄漏防护

- CheXpert 按患者划分训练、验证和内部测试，读取清单时再次检查患者集合不相交。
- MIMIC 清单包含固定种子、选择规则、相对路径、文件大小和 SHA-256 锁文件。
- 外部评价先验证清单和全部文件，再打开标签 CSV；MIMIC 不参与早停、阈值选择、模型选择或超参数调整。
- `failure_cases.local.json` 可包含受控检查标识，只能留在受控本地运行目录；`.gitignore` 明确排除它。
- `.medexperiment` 拒绝常见患者字段、影像、CSV、Parquet 和患者级预测。

## 应用准入与去标识化

### DICOM

- 只接受 `DX`/`CR`、胸部、单帧、单通道、成人 AP/PA 且可解码的影像。
- 删除非白名单数据元素和全部私有标签，移除患者姓名、患者号、检查号和原 Study ID。
- 重建最小文件元信息并生成新的 SOP、Study、Series UID。
- 对人工矩形区域直接改写像素；改写后使用未压缩 Explicit VR Little Endian 派生文件。
- `BurnedInAnnotation=YES` 时至少需要一个有效、与图像相交的遮挡矩形。

### PNG/JPEG

- 必须人工确认胸片、成人、AP/PA 体位并检查烧录文字。
- 解码为灰度像素并重新编码为 PNG，不复制 EXIF 或源元数据。
- 严重裁切、分辨率过低、损坏或无法解码的文件拒绝入库。

自动元数据清除不能证明影像绝对匿名。烧录文字检查仍是人工责任，研究和竞赛使用前还应按所在机构流程复核。

## 本地存储与保留

- Electron 用户数据目录下保存 SQLite、去标识化派生影像、模型包和实验包。
- 数据库只使用随机 `ST-*`、`PRD-*`、`REV-*`、`RPT-*`、`USR-*` 标识，不设计姓名或病历号字段。
- 临时文件在取消、拒绝、提交、失败及下次服务启动时清理。
- 卸载是否保留用户数据必须在干净 Windows 验证中明确选择并记录；未经验证前不要宣称卸载会安全删除或保留数据。
- 备份必须由使用机构在本地受控介质完成，不得把运行目录提交到 Git 或同步到未经批准的云盘。
