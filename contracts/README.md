# 共享协议

`labels.json` 是唯一的 CheXpert 14 项标签来源。模型清单、预测、复核和实验包都必须通过对应 JSON Schema 校验。

协议有意不包含姓名、病历号、原始路径、去标识化文件路径、MIMIC 患者标识或患者级预测。`deidentifiedPath` 与导入哈希只在本地 SQLite 记录中使用，不进入渲染进程或实验包。
