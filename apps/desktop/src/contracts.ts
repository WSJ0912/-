import Ajv2020, { type ErrorObject, type ValidateFunction } from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import {
  EXPERIMENT_BUNDLE_SCHEMA,
  MODEL_MANIFEST_SCHEMA,
  PREDICTION_CAM_SCHEMA,
  PREDICTION_SCHEMA,
  REPORT_REVISION_SCHEMA,
  REVIEW_SCHEMA,
  STUDY_SCHEMA,
} from "./generated/contracts";
import type {
  ExperimentBundle,
  ModelManifest,
  Prediction,
  PredictionCam,
  ReportRevision,
  Review,
  Study,
} from "./generated/contracts";

export {
  CHEXPERT_LABELS,
  LABEL_DEFINITIONS,
  LABEL_DICTIONARY_VERSION,
} from "./generated/contracts";
export type {
  CheXpertLabel,
  ConfigValue,
  CurvePoint,
  CurveSeries,
  ExperimentBundle,
  MetricInterval,
  MetricValue,
  ModelCardValue,
  ModelManifest,
  Prediction,
  PredictionCam,
  ReportRevision,
  Review,
  ReviewDecision,
  Study,
} from "./generated/contracts";

const ajv = new Ajv2020({ allErrors: true, strict: true, allowUnionTypes: true });
addFormats(ajv);

function compile<T>(schema: object): ValidateFunction<T> {
  return ajv.compile<T>(schema);
}

const experimentBundleValidator = compile<ExperimentBundle>(EXPERIMENT_BUNDLE_SCHEMA);
const modelManifestValidator = compile<ModelManifest>(MODEL_MANIFEST_SCHEMA);
const predictionCamValidator = compile<PredictionCam>(PREDICTION_CAM_SCHEMA);
const predictionValidator = compile<Prediction>(PREDICTION_SCHEMA);
const reportRevisionValidator = compile<ReportRevision>(REPORT_REVISION_SCHEMA);
const reviewValidator = compile<Review>(REVIEW_SCHEMA);
const studyValidator = compile<Study>(STUDY_SCHEMA);

export function validateExperimentBundle(value: unknown): value is ExperimentBundle {
  return experimentBundleValidator(value);
}

export function validateModelManifest(value: unknown): value is ModelManifest {
  return modelManifestValidator(value);
}

export function validatePrediction(value: unknown): value is Prediction {
  return predictionValidator(value);
}

export function validatePredictionCam(value: unknown): value is PredictionCam {
  return predictionCamValidator(value);
}

export function validateReportRevision(value: unknown): value is ReportRevision {
  return reportRevisionValidator(value);
}

export function validateReview(value: unknown): value is Review {
  return reviewValidator(value);
}

export function validateStudy(value: unknown): value is Study {
  return studyValidator(value);
}

export const contractValidators = {
  experimentBundle: experimentBundleValidator,
  modelManifest: modelManifestValidator,
  prediction: predictionValidator,
  predictionCam: predictionCamValidator,
  reportRevision: reportRevisionValidator,
  review: reviewValidator,
  study: studyValidator,
} as const;

export type ContractValidatorName = keyof typeof contractValidators;

export function contractValidationErrors(name: ContractValidatorName): readonly ErrorObject[] {
  return contractValidators[name].errors ?? [];
}
