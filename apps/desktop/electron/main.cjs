const { app, BrowserWindow, dialog, ipcMain, safeStorage } = require("electron");
const { spawn } = require("node:child_process");
const { randomUUID } = require("node:crypto");
const { existsSync, readFileSync, writeFileSync, mkdirSync } = require("node:fs");
const path = require("node:path");
const { rendererRouteAllowed } = require("./ipc-policy.cjs");

let mainWindow;
let serviceProcess;
let serviceInfo = null;
let serviceError = null;
let sessionToken = null;
let sessionRole = null;
const selectedPackages = new Map();

function serviceRoot() {
  return path.join(app.getPath("userData"), "medical-imaging-data");
}

function serviceLaunch() {
  const configured = process.env.MEDICAL_SERVICE_EXECUTABLE;
  if (configured) return { executable: configured, args: [] };
  if (app.isPackaged) {
    return {
      executable: path.join(process.resourcesPath, "inference-service", "inference-service.exe"),
      args: [],
    };
  }
  const projectRoot = path.resolve(__dirname, "..", "..", "..");
  const pythonCandidates = [
    path.join(projectRoot, ".venv", "python.exe"),
    path.join(projectRoot, ".venv", "Scripts", "python.exe"),
  ];
  const python = pythonCandidates.find((candidate) => existsSync(candidate)) || pythonCandidates[1];
  return {
    executable: python,
    args: ["-m", "inference_service.cli"],
    env: {
      ...process.env,
      PYTHONPATH: [
        path.join(projectRoot, "ml"),
        path.join(projectRoot, "services", "inference"),
        process.env.PYTHONPATH || "",
      ].filter(Boolean).join(path.delimiter),
    },
  };
}

function startLocalService() {
  const launch = serviceLaunch();
  if (!existsSync(launch.executable)) {
    serviceError = app.isPackaged
      ? "安装包中的本地推理服务缺失"
      : "开发环境缺少 .venv 中的 Python 3.11；请先安装项目依赖";
    return;
  }
  mkdirSync(serviceRoot(), { recursive: true });
  const args = [...launch.args, "--root", serviceRoot(), "--port", "0"];
  serviceProcess = spawn(launch.executable, args, {
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
    env: launch.env || process.env,
  });
  let output = "";
  serviceProcess.stdout.on("data", (chunk) => {
    output += chunk.toString();
    let newline = output.indexOf("\n");
    while (newline >= 0 && !serviceInfo) {
      const line = output.slice(0, newline).trim();
      output = output.slice(newline + 1);
      try {
        const candidate = JSON.parse(line);
        if (
          candidate.host === "127.0.0.1" &&
          Number.isInteger(candidate.port) &&
          candidate.port > 0 &&
          typeof candidate.processToken === "string" &&
          candidate.processToken.length >= 32
        ) {
          serviceInfo = candidate;
          serviceError = null;
        }
      } catch (_) {
        // Ignore non-handshake output until the service emits its JSON line.
      }
      newline = output.indexOf("\n");
    }
  });
  serviceProcess.on("error", (error) => {
    serviceError = `本地推理服务启动失败：${error.message}`;
  });
  serviceProcess.on("exit", (code) => {
    if (!app.isQuitting && code !== 0 && !serviceError) {
      serviceError = `本地推理服务意外退出（代码 ${code ?? "未知"}）`;
    }
    serviceInfo = null;
  });
}

function serviceUrl() {
  if (!serviceInfo) throw new Error("本地服务尚未启动");
  return `http://${serviceInfo.host}:${serviceInfo.port}`;
}

function keyFile() { return path.join(app.getPath("userData"), "assistant-key.bin"); }

function saveAssistantKey(value) {
  if (!safeStorage.isEncryptionAvailable()) throw new Error("系统安全存储不可用");
  mkdirSync(path.dirname(keyFile()), { recursive: true });
  writeFileSync(keyFile(), safeStorage.encryptString(value));
  return true;
}

function readAssistantKey() {
  if (!existsSync(keyFile()) || !safeStorage.isEncryptionAvailable()) return "";
  return safeStorage.decryptString(readFileSync(keyFile()));
}

function loadAssistantKey() {
  return Boolean(readAssistantKey());
}

async function requestServiceTrusted(endpoint, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  if (typeof endpoint !== "string" || !endpoint.startsWith("/")) throw new Error("无效的服务命令");
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (serviceInfo?.processToken) headers.Authorization = `Bearer ${serviceInfo.processToken}`;
  if (sessionToken) headers["X-Session-Token"] = sessionToken;
  if (endpoint.startsWith("/api/assistant/")) {
    const key = readAssistantKey();
    if (key) headers["X-Assistant-Key"] = key;
  }
  const response = await fetch(`${serviceUrl()}${endpoint}`, { ...options, method, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `本地服务错误 (${response.status})`);
  return body;
}

async function requestServiceFromRenderer(endpoint, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  if (!rendererRouteAllowed(endpoint, method)) {
    throw new Error("不允许的服务命令");
  }
  return requestServiceTrusted(endpoint, { ...options, method });
}

function registerIpc() {
  ipcMain.handle("app:status", () => ({ service: serviceInfo ? { connected: true } : null, serviceError, assistantKeyConfigured: loadAssistantKey() }));
  ipcMain.handle("session:logout", async () => {
    try {
      if (sessionToken) await requestServiceTrusted("/api/logout", { method: "POST" });
    } finally {
      sessionToken = null;
      sessionRole = null;
      selectedPackages.clear();
    }
    return true;
  });
  ipcMain.handle("command:stage-images", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile", "multiSelections"],
      filters: [{ name: "医学影像", extensions: ["dcm", "dicom", "png", "jpg", "jpeg"] }],
    });
    if (result.canceled || result.filePaths.length === 0) return [];
    return requestServiceTrusted("/api/import/stage", {
      method: "POST",
      body: JSON.stringify({ paths: result.filePaths }),
    });
  });
  ipcMain.handle("command:select-model", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile"],
      filters: [{ name: "模型包", extensions: ["medmodel"] }],
    });
    if (result.canceled || result.filePaths.length !== 1) return null;
    const selectionId = randomUUID();
    selectedPackages.set(selectionId, { kind: "model", path: result.filePaths[0] });
    return { selectionId, name: path.basename(result.filePaths[0]) };
  });
  ipcMain.handle("command:install-model", async (_, selectionId) => {
    const selected = selectedPackages.get(String(selectionId));
    if (!selected || selected.kind !== "model") throw new Error("模型包选择已失效");
    selectedPackages.delete(String(selectionId));
    return requestServiceTrusted("/api/models/install", {
      method: "POST",
      body: JSON.stringify({ packagePath: selected.path }),
    });
  });
  ipcMain.handle("command:import-experiment", async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ["openFile"],
      filters: [{ name: "实验包", extensions: ["medexperiment"] }],
    });
    if (result.canceled || result.filePaths.length !== 1) return null;
    return requestServiceTrusted("/api/experiments/import", {
      method: "POST",
      body: JSON.stringify({ packagePath: result.filePaths[0] }),
    });
  });
  ipcMain.handle("command:export-report", async (_, request) => {
    const suggestedName = path.basename(String(request?.suggestedName || "阅片报告.pdf"));
    const result = await dialog.showSaveDialog(mainWindow, {
      defaultPath: suggestedName,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });
    if (result.canceled || !result.filePath) return { exported: false };
    await requestServiceTrusted("/api/reports/export", {
      method: "POST",
      body: JSON.stringify({
        reportId: request.reportId,
        revision: request.revision,
        outputPath: result.filePath,
      }),
    });
    return { exported: true };
  });
  ipcMain.handle("service:request", async (_, request) => {
    if (!request || typeof request !== "object") throw new Error("无效的服务命令");
    const method = String(request.method || "GET").toUpperCase();
    const result = await requestServiceFromRenderer(request.endpoint, { method, body: request.body === undefined ? undefined : JSON.stringify(request.body) });
    if (result.sessionToken) {
      sessionToken = result.sessionToken;
      sessionRole = result.role;
      const { sessionToken: _, ...publicResult } = result;
      return publicResult;
    }
    return result;
  });
  ipcMain.handle("assistant:key:save", (_, value) => {
    if (sessionRole !== "admin") throw new Error("只有管理员可以修改助手设置");
    return saveAssistantKey(String(value));
  });
  ipcMain.handle("assistant:key:status", () => ({ configured: loadAssistantKey() }));
}

function createWindow() {
  mainWindow = new BrowserWindow({ width: 1440, height: 900, minWidth: 1120, minHeight: 720, backgroundColor: "#0c171d", webPreferences: { preload: path.join(__dirname, "preload.cjs"), contextIsolation: true, nodeIntegration: false, sandbox: true } });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  mainWindow.webContents.on("will-navigate", (event, targetUrl) => {
    const currentUrl = mainWindow.webContents.getURL();
    if (targetUrl !== currentUrl) event.preventDefault();
  });
  if (!app.isPackaged) mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL || "http://127.0.0.1:5173");
  else mainWindow.loadFile(path.join(__dirname, "..", "renderer-dist", "index.html"));
}

app.whenReady().then(() => { startLocalService(); registerIpc(); createWindow(); app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); }); });
app.on("before-quit", () => { app.isQuitting = true; if (serviceProcess) serviceProcess.kill(); });
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
