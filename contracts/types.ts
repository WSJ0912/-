export const CHEXPERT_LABELS = [
  "Atelectasis",
  "Cardiomegaly",
  "Consolidation",
  "Edema",
  "Enlarged Cardiomediastinum",
  "Fracture",
  "Lung Lesion",
  "Lung Opacity",
  "No Finding",
  "Pleural Effusion",
  "Pleural Other",
  "Pneumonia",
  "Pneumothorax",
  "Support Devices",
] as const;

export type CheXpertLabel = (typeof CHEXPERT_LABELS)[number];
export type Role = "admin" | "doctor";
export type ReviewDecision = "confirmed" | "denied" | "uncertain";

export interface Study {
  studyId: string;
  createdAt: string;
  modality: "DX" | "CR" | "CT" | "UNKNOWN";
  bodyPart: "CHEST" | "UNKNOWN";
  viewPosition: "AP" | "PA" | "UNKNOWN";
  adultConfirmed: boolean;
  status: "staged" | "ready" | "rejected" | "reviewed";
}

export interface ModelManifest {
  schemaVersion: "medmodel-1";
  modelId: string;
  version: string;
  architecture: "densenet121";
  labels: readonly CheXpertLabel[];
  input: { width: 320; height: 320; channels: 1; normalization: "imagenet-gray" };
  thresholds: Record<CheXpertLabel, number>;
  preprocessing: { viewPositions: ["AP", "PA"]; adultOnly: true };
  training: { commit: string; seeds: number[]; method: "baseline" | "mixstyle" };
  dataSources: string[];
  license: string;
  files: Record<string, string>;
  modelSha256: string;
  manifestSha256: string;
}

export interface Prediction {
  predictionId: string;
  studyId: string;
  modelId: string;
  modelVersion: string;
  modelSha256: string;
  manifestSha256: string;
  createdAt: string;
  probabilities: Record<CheXpertLabel, number | null>;
  logits: Record<CheXpertLabel, number | null>;
  thresholds: Partial<Record<CheXpertLabel, number>>;
  camAvailable: boolean;
  source: "onnx";
}

export interface Review {
  reviewId: string;
  studyId: string;
  predictionId: string;
  doctorId: string;
  createdAt: string;
  decisions: Partial<Record<CheXpertLabel, ReviewDecision>>;
  notes: string;
}

export interface ReportRevision {
  reportId: string;
  studyId: string;
  revision: number;
  authorId: string;
  createdAt: string;
  body: string;
  status: "draft" | "confirmed";
  confirmedAt?: string;
  reviewId?: string;
}

export interface ExperimentBundle {
  schemaVersion: "medexperiment-1";
  experimentId: string;
  config: Record<string, unknown>;
  seeds: number[];
  datasetManifestSha256: string;
  aggregateMetrics: Record<string, number | null>;
  perClassMetrics: Record<string, Record<string, number | null>>;
  curves: Record<string, unknown>;
  modelCard: Record<string, unknown>;
}
