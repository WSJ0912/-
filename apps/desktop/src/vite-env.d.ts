/// <reference types="vite/client" />

interface Window {
  medicalApi: {
    appStatus(): Promise<{ service: { connected: true } | null; serviceError?: string | null; assistantKeyConfigured: boolean }>;
    logout(): Promise<boolean>;
    stageSelectedImages(): Promise<any[]>;
    selectModelPackage(): Promise<{ selectionId: string; name: string } | null>;
    installSelectedModel(selectionId: string): Promise<any>;
    importExperiment(): Promise<any | null>;
    exportReport(reportId: string, revision: number, suggestedName?: string): Promise<{ exported: boolean }>;
    request(endpoint: string, method?: string, body?: unknown): Promise<any>;
    saveAssistantKey(value: string): Promise<boolean>;
    assistantKeyStatus(): Promise<{ configured: boolean }>;
  };
}
