export type Role = "admin" | "doctor";
export type NavKey = "queue" | "workbench" | "reports" | "experiments" | "models" | "admin";

export const labels = [
  ["Atelectasis", "肺不张"], ["Cardiomegaly", "心脏增大"], ["Consolidation", "实变"], ["Edema", "水肿"],
  ["Enlarged Cardiomediastinum", "纵隔增宽"], ["Fracture", "骨折"], ["Lung Lesion", "肺部病灶"], ["Lung Opacity", "肺部阴影"],
  ["No Finding", "未见异常"], ["Pleural Effusion", "胸腔积液"], ["Pleural Other", "其他胸膜病变"], ["Pneumonia", "肺炎"],
  ["Pneumothorax", "气胸"], ["Support Devices", "支持性器械"],
] as const;

export async function request(endpoint: string, method = "GET", body?: unknown) {
  return window.medicalApi.request(endpoint, method, body);
}

export function formatProbability(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--";
}
