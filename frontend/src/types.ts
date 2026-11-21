export type FieldDefinition = {
  key: string;
  label: string;
  required: boolean;
  hint?: string;
} & (
  | { type: 'number'; unit: string; min?: number; max?: number; integer?: boolean }
  | { type: 'select'; options: { value: string; label: string }[] }
);

export interface ModelInfo {
  indicatorName: string;
  unit: string;
  // null preserves the server value without rounding.
  decimalPlaces: number | null;
  fields: FieldDefinition[];
  modelVersion?: string;
  predictionType?: 'estimated_current_measurement';
  demo?: boolean;
}

export type FormValues = Record<string, string>;
export type FieldErrors = Record<string, string>;
export type PredictionInput = Record<string, number | string>;

export interface PredictionInterval {
  lower: number;
  upper: number;
  /** Nominal level of the calibration, e.g. 0.9. */
  level: number;
  /** Coverage measured on the external validation cohort; marginal, not a personal guarantee. */
  measuredCoverage: number;
  method?: string;
}

export interface PredictionContribution {
  key: string;
  value: number;
}

/** Exact baseline Shapley decomposition: referenceValue + Σ contributions = predictedValue. Not causal. */
export interface PredictionExplanation {
  referenceValue: number;
  contributions: PredictionContribution[];
  method?: string;
}

export type PredictionWarning = 'OUT_OF_DISTRIBUTION';

export interface Prediction {
  predictedValue: number;
  unit: string;
  modelVersion?: string;
  disclaimer?: string;
  predictionType?: 'estimated_current_measurement';
  demo?: boolean;
  interval?: PredictionInterval;
  explanation?: PredictionExplanation;
  warnings?: PredictionWarning[];
}

export interface PredictionApi {
  getModelInfo(signal: AbortSignal, language?: 'ru' | 'kk' | 'en'): Promise<ModelInfo>;
  predict(input: PredictionInput, signal: AbortSignal, language?: 'ru' | 'kk' | 'en'): Promise<Prediction>;
}
