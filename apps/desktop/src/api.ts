import { LABEL_DEFINITIONS } from "./generated/contracts";

export type Role = "admin" | "doctor";
export type NavKey = "queue" | "workbench" | "reports" | "experiments" | "models" | "admin";

export const labels = LABEL_DEFINITIONS.map(
  ({ key, zh }) => [key, zh] as const,
);

export async function request<T = any>(endpoint: string, method = "GET", body?: unknown): Promise<T> {
  return window.medicalApi.request<T>(endpoint, method, body);
}

export function formatProbability(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--";
}
