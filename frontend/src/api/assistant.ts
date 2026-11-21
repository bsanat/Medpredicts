import type { Prediction } from '../types';

export interface AssistantSource { id: string; title: string; document: string }
export interface AssistantReply { answer: string; refused: boolean; sources: AssistantSource[]; model: string }
export interface AssistantTurn { role: 'user' | 'assistant'; text: string }
/** Derived numbers only; raw medical inputs are never part of an assistant request. */
export interface AssistantResultContext {
  predictedValue: number;
  intervalLower?: number;
  intervalUpper?: number;
  measuredCoverage?: number;
  referenceValue?: number;
  contributions?: { key: string; value: number }[];
  warnings?: string[];
}
export interface AssistantRequest {
  message: string;
  language: 'ru' | 'kk' | 'en';
  history: AssistantTurn[];
  resultContext?: AssistantResultContext;
}
export type AssistantFailure = 'unavailable' | 'rateLimited' | 'invalid' | 'network' | 'timeout' | 'unexpected';

export class AssistantError extends Error {
  constructor(public readonly kind: AssistantFailure) {
    super(kind);
    this.name = 'AssistantError';
  }
}

export interface AssistantClient {
  status(signal: AbortSignal): Promise<boolean>;
  ask(request: AssistantRequest, signal: AbortSignal): Promise<AssistantReply>;
}

export const MAX_ASSISTANT_MESSAGE = 1000;
export const MAX_ASSISTANT_HISTORY = 6;

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;

export function parseAssistantReply(value: unknown): AssistantReply {
  if (!record(value) || !text(value.answer) || typeof value.refused !== 'boolean' || !Array.isArray(value.sources)
    || typeof value.model !== 'string') throw new AssistantError('unexpected');
  const sources = value.sources.map((source) => {
    if (!record(source) || !text(source.id) || !text(source.title) || !text(source.document)) throw new AssistantError('unexpected');
    return { id: source.id, title: source.title, document: source.document };
  });
  return { answer: value.answer, refused: value.refused, sources, model: value.model };
}

export function contextFromPrediction(result: Prediction): AssistantResultContext {
  return {
    predictedValue: result.predictedValue,
    ...(result.interval ? { intervalLower: result.interval.lower, intervalUpper: result.interval.upper,
      measuredCoverage: result.interval.measuredCoverage } : {}),
    ...(result.explanation ? { referenceValue: result.explanation.referenceValue,
      contributions: result.explanation.contributions.map(({ key, value }) => ({ key, value })) } : {}),
    ...(result.warnings?.length ? { warnings: [...result.warnings] } : {}),
  };
}

export function createAssistantClient(baseUrl: string, timeoutMs: number): AssistantClient {
  async function call(path: string, signal: AbortSignal, body?: unknown): Promise<unknown> {
    const timeout = AbortSignal.timeout(timeoutMs);
    try {
      const response = await fetch(`${baseUrl.replace(/\/$/, '')}${path}`, {
        method: body === undefined ? 'GET' : 'POST',
        signal: AbortSignal.any([signal, timeout]),
        headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        credentials: 'omit', cache: 'no-store', referrerPolicy: 'no-referrer',
      });
      if (response.status === 503) throw new AssistantError('unavailable');
      if (response.status === 429) throw new AssistantError('rateLimited');
      if (response.status === 400) throw new AssistantError('invalid');
      if (!response.ok) throw new AssistantError('unexpected');
      return await response.json();
    } catch (error) {
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      if (timeout.aborted) throw new AssistantError('timeout');
      if (error instanceof AssistantError) throw error;
      throw new AssistantError(error instanceof SyntaxError ? 'unexpected' : 'network');
    }
  }
  return {
    status: async (signal) => {
      const value = await call('/api/v1/assistant/status', signal);
      return record(value) && value.available === true;
    },
    ask: async (request, signal) => parseAssistantReply(await call('/api/v1/assistant/messages', signal, {
      ...request, history: request.history.slice(-MAX_ASSISTANT_HISTORY),
    })),
  };
}
