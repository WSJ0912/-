const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("medicalApi", Object.freeze({
  appStatus: () => ipcRenderer.invoke("app:status"),
  logout: () => ipcRenderer.invoke("session:logout"),
  stageSelectedImages: () => ipcRenderer.invoke("command:stage-images"),
  selectModelPackage: () => ipcRenderer.invoke("command:select-model"),
  installSelectedModel: (selectionId) => ipcRenderer.invoke("command:install-model", selectionId),
  importExperiment: () => ipcRenderer.invoke("command:import-experiment"),
  exportReport: (reportId, revision, suggestedName) => ipcRenderer.invoke("command:export-report", { reportId, revision, suggestedName }),
  request: (endpoint, method = "GET", body) => ipcRenderer.invoke("service:request", { endpoint, method, body }),
  saveAssistantKey: (value) => ipcRenderer.invoke("assistant:key:save", value),
  assistantKeyStatus: () => ipcRenderer.invoke("assistant:key:status"),
}));
