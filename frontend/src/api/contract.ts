import type { FieldDefinition, ModelInfo, Prediction, PredictionExplanation, PredictionInterval, PredictionWarning } from '../types';
import { ru } from '../i18n/ru';
import { ApiError } from './errors';

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const optionalText = (value: unknown) => value === undefined || text(value);
const validSemantics = (value: Record<string, unknown>) =>
  (value.predictionType === undefined || value.predictionType === 'estimated_current_measurement')
  && (value.demo === undefined || typeof value.demo === 'boolean');
const unexpected = () => new ApiError('unexpected', ru.apiErrors.unexpected);

function parseInterval(value: unknown, estimate: number): PredictionInterval {
  if (!record(value) || !finite(value.lower) || !finite(value.upper) || !finite(value.level) || !finite(value.measuredCoverage)
    || value.lower > estimate || estimate > value.upper || value.level <= 0 || value.level >= 1
    || value.measuredCoverage < 0 || value.measuredCoverage > 1 || !optionalText(value.method)) throw unexpected();
  return { lower: value.lower, upper: value.upper, level: value.level, measuredCoverage: value.measuredCoverage,
    ...(value.method === undefined ? {} : { method: value.method as string }) };
}

const KNOWN_WARNINGS: readonly PredictionWarning[] = ['OUT_OF_DISTRIBUTION'];
// Server sums float32 graph outputs; allow a small accumulation error in the exact decomposition.
const ADDITIVITY_TOLERANCE = 0.05;

function parseExplanation(value: unknown, estimate: number): PredictionExplanation {
  if (!record(value) || !finite(value.referenceValue) || !Array.isArray(value.contributions) || value.contributions.length === 0
    || !optionalText(value.method)) throw unexpected();
  const contributions = value.contributions.map((item) => {
    if (!record(item) || !text(item.key) || !finite(item.value)) throw unexpected();
    return { key: item.key, value: item.value };
  });
  if (new Set(contributions.map((item) => item.key)).size !== contributions.length) throw unexpected();
  const total = contributions.reduce((sum, item) => sum + item.value, value.referenceValue);
  if (Math.abs(total - estimate) > ADDITIVITY_TOLERANCE) throw unexpected();
  return { referenceValue: value.referenceValue, contributions,
    ...(value.method === undefined ? {} : { method: value.method as string }) };
}

function parseWarnings(value: unknown): PredictionWarning[] {
  if (!Array.isArray(value) || !value.every((item) => KNOWN_WARNINGS.includes(item as PredictionWarning))) throw unexpected();
  return [...new Set(value as PredictionWarning[])];
}

export function parsePrediction(value: unknown, expectedUnit?: string): Prediction {
  if (!record(value) || !finite(value.predictedValue) || !text(value.unit)
    || !optionalText(value.modelVersion) || !optionalText(value.disclaimer) || !validSemantics(value)
    || (expectedUnit !== undefined && value.unit !== expectedUnit)) throw unexpected();
  return {
    predictedValue: value.predictedValue,
    unit: value.unit,
    ...(value.modelVersion === undefined ? {} : { modelVersion: value.modelVersion as string }),
    ...(value.disclaimer === undefined ? {} : { disclaimer: value.disclaimer as string }),
    ...(value.predictionType === undefined ? {} : { predictionType: value.predictionType as Prediction['predictionType'] }),
    ...(value.demo === undefined ? {} : { demo: value.demo as boolean }),
    ...(value.interval === undefined ? {} : { interval: parseInterval(value.interval, value.predictedValue) }),
    ...(value.explanation === undefined ? {} : { explanation: parseExplanation(value.explanation, value.predictedValue) }),
    ...(value.warnings === undefined ? {} : { warnings: parseWarnings(value.warnings) }),
  };
}

function parseField(value: unknown): FieldDefinition {
  if (!record(value) || !text(value.key) || !/^[a-zA-Z][a-zA-Z0-9_]*$/.test(value.key)
    || ['constructor', 'prototype', '__proto__'].includes(value.key)
    || !text(value.label) || typeof value.required !== 'boolean' || !optionalText(value.hint)) throw unexpected();
  const common = { key: value.key, label: value.label, required: value.required,
    ...(value.hint === undefined ? {} : { hint: value.hint as string }) };
  if (value.type === 'number') {
    if (!text(value.unit) || (value.min !== undefined && !finite(value.min))
      || (value.max !== undefined && !finite(value.max))
      || (finite(value.min) && finite(value.max) && value.min > value.max)
      || (value.integer !== undefined && typeof value.integer !== 'boolean')) throw unexpected();
    return { ...common, type: 'number', unit: value.unit,
      ...(value.min === undefined ? {} : { min: value.min as number }),
      ...(value.max === undefined ? {} : { max: value.max as number }),
      ...(value.integer === undefined ? {} : { integer: value.integer as boolean }) };
  }
  if (value.type === 'select' && Array.isArray(value.options) && value.options.length > 0) {
    const options = value.options.map((option: unknown) => {
      if (!record(option) || !text(option.value) || !text(option.label)) throw unexpected();
      return { value: option.value, label: option.label };
    });
    if (new Set(options.map((option) => option.value)).size !== options.length) throw unexpected();
    return { ...common, type: 'select', options };
  }
  throw unexpected();
}

// GET contract implemented by MedPredicts/backend; validate before enabling input.
export function parseModelInfo(value: unknown): ModelInfo {
  if (!record(value) || !text(value.indicatorName) || !text(value.unit)
    || !optionalText(value.modelVersion) || !validSemantics(value)
    || !(value.decimalPlaces === null || (finite(value.decimalPlaces)
      && Number.isInteger(value.decimalPlaces) && value.decimalPlaces >= 0 && value.decimalPlaces <= 20))
    || !Array.isArray(value.fields) || value.fields.length === 0) throw unexpected();
  const fields = value.fields.map(parseField);
  if (new Set(fields.map((field) => field.key)).size !== fields.length) throw unexpected();
  return { indicatorName: value.indicatorName, unit: value.unit,
    decimalPlaces: value.decimalPlaces, fields,
    ...(value.modelVersion === undefined ? {} : { modelVersion: value.modelVersion as string }),
    ...(value.predictionType === undefined ? {} : { predictionType: value.predictionType as ModelInfo['predictionType'] }),
    ...(value.demo === undefined ? {} : { demo: value.demo as boolean }) };
}

export function parseValidationError(value: unknown): ApiError {
  if (!record(value) || value.error !== 'VALIDATION_ERROR' || !Array.isArray(value.fields)
    || value.fields.length === 0) throw unexpected();
  const fields = value.fields.map((item: unknown) => {
    if (!record(item) || !text(item.field) || !text(item.message)) throw unexpected();
    return { field: item.field, message: item.message };
  });
  return new ApiError('validation', ru.validationSummary, fields);
}
