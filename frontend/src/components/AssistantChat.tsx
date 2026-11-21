import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { Copy } from '../i18n/ru';
import type { Language } from '../i18n';
import { configuredAssistant } from '../config/environment';
import { AssistantError, MAX_ASSISTANT_MESSAGE, type AssistantClient, type AssistantReply,
  type AssistantResultContext, type AssistantTurn } from '../api/assistant';
import { OPEN_ASSISTANT } from '../api/assistantContext';
import { Icon } from './Icon';

interface Message { role: 'user' | 'assistant'; text: string; reply?: AssistantReply }
type Availability = 'checking' | 'available' | 'unavailable' | 'demo';

export function AssistantChat({ copy, language, openFromRoute }: { copy: Copy; language: Language; openFromRoute: boolean }) {
  const [client] = useState<AssistantClient | undefined>(configuredAssistant);
  const [started, setStarted] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const text = copy.assistant;
  const [availability, setAvailability] = useState<Availability>(client ? 'checking' : 'demo');
  const [context, setContext] = useState<AssistantResultContext>();
  const [draft, setDraft] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const request = useRef<AbortController | null>(null);
  const log = useRef<HTMLDivElement>(null);

  function openChat() {
    setStarted(true);
    if (!dialog.current?.open) dialog.current?.showModal();
  }

  useEffect(() => {
    if (openFromRoute) openChat();
  }, [openFromRoute]);

  useEffect(() => {
    function openWithResult(event: Event) {
      setContext((event as CustomEvent<AssistantResultContext>).detail);
      setDraft(text.contextQuestion);
      openChat();
    }
    window.addEventListener(OPEN_ASSISTANT, openWithResult);
    return () => window.removeEventListener(OPEN_ASSISTANT, openWithResult);
  }, [text.contextQuestion]);

  useEffect(() => {
    if (!client || !started) return;
    const controller = new AbortController();
    client.status(controller.signal).then((ok) => setAvailability(ok ? 'available' : 'unavailable'))
      .catch((reason: unknown) => { if (!controller.signal.aborted) setAvailability('unavailable'); return reason; });
    return () => controller.abort();
  }, [client, started]);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => { if (log.current) log.current.scrollTop = log.current.scrollHeight; }, [messages, busy]);

  async function send(message: string) {
    const question = message.trim();
    if (!client || request.current || !question || question.length > MAX_ASSISTANT_MESSAGE) return;
    const controller = new AbortController();
    request.current = controller;
    const history: AssistantTurn[] = messages.map(({ role, text: content }) => ({ role, text: content }));
    setMessages((current) => [...current, { role: 'user', text: question }]);
    setDraft('');
    setBusy(true);
    setError(undefined);
    try {
      const reply = await client.ask({ message: question, language, history, ...(context ? { resultContext: context } : {}) },
        controller.signal);
      setMessages((current) => [...current, { role: 'assistant', text: reply.answer, reply }]);
      setContext(undefined);
    } catch (reason) {
      if (controller.signal.aborted) return;
      setError(text.errors[reason instanceof AssistantError ? reason.kind : 'unexpected']);
    } finally {
      if (request.current === controller) request.current = null;
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void send(draft);
  }

  return <>
    <button className="button button-primary assistant-launcher" type="button" onClick={openChat}
      aria-haspopup="dialog" aria-controls="assistant-chat">
      <Icon name="brain" />{copy.assistantNav}
    </button>
    <dialog ref={dialog} id="assistant-chat" className="assistant-chat" aria-labelledby="assistant-heading"
      onClose={() => { if (context) { setContext(undefined); setDraft(''); } }}>
      <header className="assistant-header">
        <Icon name="brain" /><h2 id="assistant-heading">{text.title}</h2>
        <button className="button assistant-close" type="button" aria-label={text.close}
          onClick={() => dialog.current?.close()}><Icon name="close" /></button>
      </header>
      <div className="assistant-body" ref={log}>
        <p>{text.intro}</p>
        <p className="assistant-privacy" role="note"><Icon name="lock" /><span>{text.privacy}</span></p>
        {availability === 'checking' && <p role="status">{text.checking}</p>}
        {availability === 'demo' && <p className="assistant-unavailable" role="status">{text.demoUnavailable}</p>}
        {availability === 'unavailable' && <p className="assistant-unavailable" role="status">{text.unavailable}</p>}
        {availability === 'available' && <>
          <div className="assistant-log" aria-live="polite">
            {messages.length === 0 && <div className="assistant-suggestions">
              {text.suggestions.map((suggestion) => <button key={suggestion} type="button" className="button button-secondary"
                onClick={() => void send(suggestion)} disabled={busy}>{suggestion}</button>)}
            </div>}
            {messages.map((message, index) => <article key={index} className={`assistant-message ${message.role}`}>
              <h2 className="assistant-author">{message.role === 'user' ? text.you : text.bot}</h2>
              {message.reply?.refused && <p className="assistant-refused">{text.refused}</p>}
              <p className="assistant-text">{message.text}</p>
              {message.reply && message.reply.sources.length > 0 && <details className="assistant-sources">
                <summary>{text.sources} ({message.reply.sources.length})</summary>
                <ul>{message.reply.sources.map((source) => <li key={source.id}>{source.title} — <code>{source.document}</code></li>)}</ul>
              </details>}
            </article>)}
            {busy && <p className="assistant-busy" role="status"><span className="spinner" aria-hidden="true" />{text.sending}</p>}
          </div>
          {error && <p className="assistant-error" role="alert">{error}</p>}
        </>}
        </div>
        {availability === 'available' && <div className="assistant-compose">
          {context && <div className="assistant-context" role="note"><p>{text.contextNotice}</p>
            <button type="button" className="button button-secondary" onClick={() => { setContext(undefined); setDraft(''); }}>{text.removeContext}</button>
          </div>}
          <form className="assistant-form" onSubmit={submit}>
            <label className="sr-only" htmlFor="assistant-input">{text.placeholder}</label>
            <textarea id="assistant-input" value={draft} maxLength={MAX_ASSISTANT_MESSAGE} rows={2} placeholder={text.placeholder}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(draft); } }} />
            <div className="assistant-actions">
              <span className="assistant-counter" aria-live="off">{text.counter(draft.length, MAX_ASSISTANT_MESSAGE)}</span>
              <button className="button button-primary" type="submit" disabled={busy || !draft.trim()}>{text.send}</button>
            </div>
          </form>
        </div>}
    </dialog>
  </>;
}
