# 共享协议

`labels.json` 是 CheXpert 14 项标签和中英文元数据的规范源，`schemas/*.schema.json` 是交换对象的规范源。不要手工维护 Python 或 TypeScript 中的标签、接口或 Schema 副本。

从仓库根目录生成两端代码：

```powershell
python scripts/generate-contracts.py
python scripts/generate-contracts.py --check
```

生成器会更新 `ml/cxr_research/labels.py`、`contracts/types.ts` 和 `apps/desktop/src/generated/contracts.ts`。`--check` 不写文件，任一生成产物缺失或与规范源不一致时返回非零状态，适合在 CI 中执行。

Python 制品读写使用 Draft 2020-12 校验后再执行哈希、路径和患者级字段等语义检查。桌面端使用同一批内嵌 Schema 的 Ajv 2020 校验器，不依赖运行时读取仓库文件。

协议有意不包含姓名、病历号、原始路径、去标识化文件路径、MIMIC 患者标识或患者级预测。`deidentifiedPath` 与导入哈希只在本地 SQLite 记录中使用，不进入渲染进程或实验包。
