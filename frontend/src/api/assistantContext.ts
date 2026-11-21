import type { AssistantResultContext } from './assistant';

export const OPEN_ASSISTANT = 'medpredict:open-assistant';

export function openAssistantWithResult(context: AssistantResultContext) {
  window.dispatchEvent(new CustomEvent(OPEN_ASSISTANT, { detail: context }));
}
