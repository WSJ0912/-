# 胸片多标签研究协议

## 研究问题

比较相同训练条件下的 ImageNet 初始化 DenseNet-121 基线与唯一主要改进 MixStyle，评估 MixStyle 是否提高成人 AP/PA 胸片在外部域上的多标签筛查表现。该研究不证明临床有效性。

## 固定协议

- 输入：单通道 `320 x 320`。
- 输出：固定顺序的 CheXpert 14 项观察。
- 不确定标签：目标占位为 0、mask 为 0，不作为阴性参与损失或指标。
- 损失：masked BCE；按训练集已观察标签计算逆患病率权重，上限为 10。
- 基线：DenseNet-121，ImageNet 权重的首层按 RGB 通道均值转换为单通道。
- 改进：只加入 MixStyle，`p=0.5`、`alpha=0.1`，位于前两个 DenseNet 特征阶段之后。
- 正式种子：`17`、`29`、`43`；两种方法各 3 次，共最多 6 次完整训练。
- 每次最多 15 轮，按 CheXpert 验证集五项主要异常 macro AUROC 早停，默认 patience 为 3。
- smoke test 关闭 ImageNet 权重并标记 `formal=false`，不得进入正式实验包或部署导出。

## 标签分组

- 主要五项：Atelectasis、Cardiomegaly、Consolidation、Edema、Pleural Effusion。
- 补充七项异常：Enlarged Cardiomediastinum、Fracture、Lung Lesion、Lung Opacity、Pleural Other、Pneumonia、Pneumothorax。
- 单独报告：No Finding、Support Devices。

所有 14 项均报告 AUROC、AUPRC、敏感度、特异度、F1、Brier score，以及上述每项指标的患者级 percentile bootstrap 95% 置信区间。阈值只在 CheXpert 验证集上选择，目标敏感度约 90%。指标不可计算时使用 JSON `null`，UI 显示 `--`。

## 数据准备

```powershell
python -m cxr_research prepare-chexpert `
  D:\datasets\CheXpert-v1.0-small\train.csv `
  data\chexpert-split.json `
  --image-root D:\datasets

python -m cxr_research formal-plan --output runs\formal-plan.json
```

清单包含源 CSV SHA-256、患者级划分、影像字节数和清单哈希。任何患者交叉或清单修改都会在读取时失败。

## 六次正式训练

对 `baseline` 和 `mixstyle` 分别以三个固定种子执行：

```powershell
python -m cxr_research train data\chexpert-split.json runs\mixstyle-17 `
  --method mixstyle --seed 17 --device cuda
```

正式 GPU 运行开始前记录供应商、实例、开始/结束时间和费用。六次完整训练总租用预算不得超过 300 元；超预算时停止，不得通过删改不利结果来满足预算。

## 部署选择与 ONNX

```powershell
python -m cxr_research select-deployment `
  runs\mixstyle-17\result.json `
  runs\mixstyle-29\result.json `
  runs\mixstyle-43\result.json `
  --output runs\deployment-selection.json

python -m cxr_research export-onnx `
  runs\mixstyle-29\best.pt artifacts\deployment `
  --selection runs\deployment-selection.json `
  --thresholds runs\mixstyle-29\thresholds.json `
  --model-id cxr-mixstyle --version 0.1.0 `
  --training-commit <git-commit> `
  --data-source CheXpert-v1.0-small `
  --license <verified-weight-license> `
  --package artifacts\cxr-mixstyle-0.1.0.medmodel
```

示例中的 checkpoint 路径必须替换为选择记录中的真实最优候选。选择记录固定三份结果和 checkpoint 的 SHA-256；导出拒绝非最优候选、baseline、smoke checkpoint 或被修改的文件。ONNX 必须输出 `[N,14]` logits、最后特征图和由分类器权重计算的 `[N,14,H,W]` 逐类 CAM，PyTorch/ONNX 概率最大误差必须不超过 `1e-4`。

## MIMIC 封存外部评价

```powershell
python -m cxr_research sample-mimic metadata.csv data\mimic-sealed.json `
  --image-root D:\datasets\mimic-cxr-jpg `
  --chexpert-manifest data\chexpert-split.json `
  --target-patients 8000

python -m cxr_research verify-mimic data\mimic-sealed.json

python -m cxr_research evaluate-mimic `
  data\mimic-sealed.json labels.csv artifacts\cxr-mixstyle-0.1.0.medmodel `
  runs\mimic-final --image-root D:\datasets\mimic-cxr-jpg
```

外部结果只输出聚合指标；含检查标识的失败案例写入本地排除文件。首次查看 MIMIC 结果后不得再调整模型、阈值、样本清单或预处理。

## 当前实证状态

仓库已经实现并用合成数据验证研究协议、真实 DenseNet ONNX 导出和 ONNX Runtime 路径，但尚未执行真实 CheXpert 训练、六次正式 GPU 实验或 MIMIC 外部评价。因此仓库中没有真实性能数字，也不能据此宣称筛查准确率得到提升。
