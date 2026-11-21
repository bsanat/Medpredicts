import { afterEach, describe, expect, it, vi } from 'vitest';
import { AssistantError, contextFromPrediction, createAssistantClient, parseAssistantReply } from './assistant';
import { setPendingAssistantContext, takePendingAssistantContext } from './assistantContext';

afterEach(() => vi.unstubAllGlobals());
const signal = () => new AbortController().signal;
const reply = { answer: 'Интервал…', refused: false, sources: [{ id: 'interval', title: 'Интервал', document: 'ml-ai/x.md' }], model: 'gemini' };

describe('assistant contract', () => {
  it('parses a grounded reply and rejects malformed ones', () => {
    expect(parseAssistantReply(reply)).toEqual(reply);
    for (const bad of [null, { ...reply, answer: '' }, { ...reply, refused: 'no' }, { ...reply, sources: [{ id: 'x' }] }, { ...reply, model: 1 }]) {
      expect(() => parseAssistantReply(bad)).toThrow(AssistantError);
    }
  });

  it('builds result context from derived numbers only', () => {
    const context = contextFromPrediction({ predictedValue: 125.3, unit: 'мм рт. ст.',
      interval: { lower: 99.8, upper: 150.8, level: 0.9, measuredCoverage: 0.913 },
      explanation: { referenceValue: 122.7, contributions: [{ key: 'age', value: 1.4 }, { key: 'sex', value: 1.2 }] },
      warnings: ['OUT_OF_DISTRIBUTION'] });
    expect(context).toEqual({ predictedValue: 125.3, intervalLower: 99.8, intervalUpper: 150.8, measuredCoverage: 0.913,
      referenceValue: 122.7, contributions: [{ key: 'age', value: 1.4 }, { key: 'sex', value: 1.2 }], warnings: ['OUT_OF_DISTRIBUTION'] });
    expect(Object.keys(context)).not.toContain('unit');
  });

  it('keeps a pending context in memory for exactly one read', () => {
    setPendingAssistantContext({ predictedValue: 120 });
    expect(takePendingAssistantContext()).toEqual({ predictedValue: 120 });
    expect(takePendingAssistantContext()).toBeUndefined();
  });

  it('posts to the backend with trimmed history and maps HTTP failures', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(reply), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = createAssistantClient('http://api.test/', 1000);
    const history = Array.from({ length: 9 }, (_, i) => ({ role: i % 2 ? 'assistant' as const : 'user' as const, text: `t${i}` }));
    await expect(client.ask({ message: 'Q', language: 'kk', history }, signal())).resolves.toEqual(reply);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit & { body: string }];
    expect(url).toBe('http://api.test/api/v1/assistant/messages');
    expect(init.credentials).toBe('omit');
    const body = JSON.parse(init.body);
    expect(body.history).toHaveLength(6);
    expect(body.language).toBe('kk');
    for (const [status, kind] of [[503, 'unavailable'], [429, 'rateLimited'], [400, 'invalid'], [500, 'unexpected']] as const) {
      fetchMock.mockResolvedValueOnce(new Response('{}', { status }));
      await expect(client.ask({ message: 'Q', language: 'ru', history: [] }, signal())).rejects.toMatchObject({ kind });
    }
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ available: true }), { status: 200 }));
    await expect(client.status(signal())).resolves.toBe(true);
  });
});
