# Windows 开发与构建

## 开发环境

- Windows 10/11 x64
- Python 3.11 x64
- Node.js 20 或 22
- Git

```powershell
git clone https://github.com/WSJ0912/-.git D:\医疗影像识别
cd D:\医疗影像识别
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml,service,build]"
cd apps\desktop
npm ci
```

开发启动器同时识别标准 venv 的 `.venv\Scripts\python.exe` 和 Conda 风格的 `.venv\python.exe`。

## 验证

```powershell
cd D:\医疗影像识别
.\.venv\Scripts\python.exe -m ruff check ml services tests\python scripts
.\.venv\Scripts\python.exe -m compileall -q ml services tests\python scripts
.\.venv\Scripts\python.exe -m unittest discover -s tests\python -v

cd apps\desktop
node --check electron\main.cjs
node --check electron\preload.cjs
npm run typecheck
npm run build
npm audit --audit-level=low
```

## 开发运行

```powershell
cd D:\医疗影像识别\apps\desktop
npm run dev
```

Vite 只监听 `127.0.0.1:5173`。Electron 另行启动 Python 服务，服务在标准输出发出随机端口握手。浏览器直接打开 Vite 只用于无 preload 的视觉预览，完整功能必须从 Electron 运行。

## 构建安装包

```powershell
cd D:\医疗影像识别\apps\desktop
npm run dist
```

构建顺序为：

1. 用项目 Python 3.11 和 PyInstaller 生成 `services\inference\dist\inference-service\` 目录式运行时。
2. TypeScript 类型检查并构建 React 静态资源。
3. electron-builder 生成 Windows x64 NSIS 安装包。

Vite 渲染产物输出到 `apps\desktop\renderer-dist`，electron-builder 安装产物输出到 `apps\desktop\release`，两者必须分离以避免把旧安装包递归打入 `app.asar`。electron-builder 复用 `npm ci` 已安装且由锁文件约束的 `node_modules\electron\dist`，不会在每次打包时重复下载 Electron。安装包包含本地推理服务、应用源码包、Apache-2.0 许可证和 `third_party` 依赖声明；不包含模型权重、影像、数据库或 OpenAI 密钥。

构建脚本会把所选 Python 环境的运行库目录置于 DLL 搜索路径首位，并核验打包后的 OpenSSL DLL SHA-256 与该环境一致，防止误收集宿主机其他 Conda 环境的 DLL。

目标机器无需安装 Python、Node 或 NVIDIA 驱动。推理固定使用 ONNX Runtime CPU provider。

## 打包后服务检查

可直接运行打包服务并读取首行 JSON 握手：

```powershell
$runtime = Join-Path $env:TEMP "medical-imaging-service-check"
.\services\inference\dist\inference-service\inference-service.exe --root $runtime --port 0
```

使用握手中的随机令牌访问 `/health`，然后验证管理员创建、合成 `.medmodel` 安装、ONNX 推理和中文 PDF。测试结束后停止进程并删除该专用临时目录。

## 发布前仍需人工完成

- 在没有 Python/Node 的干净 Windows 10 x64 和 Windows 11 x64 上安装、启动、CPU 推理、卸载。
- 分别验证“保留本地数据”和计划中的清理行为，不根据开发机结果推断。
- 在 Ryzen 5 5600 上用最终许可模型实测单张推理并记录中位数、P95 和冷启动时间；目标不超过 15 秒。
- 使用代码签名证书签名安装包并检查 SmartScreen 行为。
- 确认最终模型权重的再分发许可后，才能把权重作为独立资产发布。
