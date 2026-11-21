import type { ModelInfo, PredictionExplanation } from '../types';
import type { Copy } from '../i18n/ru';
import type { Language } from '../i18n';
import { formatPrediction, signed } from '../validation';

interface Props {
  explanation: PredictionExplanation;
  model: ModelInfo;
  unit: string;
  copy: Copy;
  language: Language;
}

/** Diverging bars for exact Shapley contributions, largest absolute effect first. */
export function ContributionChart({ explanation, model, unit, copy, language }: Props) {
  const labels = new Map(model.fields.map((field) => [field.key, field.label]));
  const sorted = [...explanation.contributions].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const scale = Math.max(1, ...sorted.map((item) => Math.abs(item.value)));
  return <section className="contributions" aria-labelledby="contributions-heading">
    <h4 id="contributions-heading">{copy.explanationTitle}</h4>
    <p className="contributions-reference">{copy.explanationReference(formatPrediction(explanation.referenceValue, 1, language), unit)}</p>
    <ul className="contribution-list">
      {sorted.map((item) => {
        const width = `${(Math.abs(item.value) / scale) * 50}%`;
        const direction = item.value < 0 ? 'negative' : 'positive';
        return <li key={item.key} className="contribution-row">
          <span className="contribution-label">{labels.get(item.key) ?? item.key}</span>
          <span className="contribution-track" aria-hidden="true">
            <span className={`contribution-bar ${direction}`} style={{ width }} />
          </span>
          <span className="contribution-value">{signed(item.value, language)} {unit}</span>
        </li>;
      })}
    </ul>
    <p className="contributions-note">{copy.explanationNote}</p>
  </section>;
}
