# MedPredict workspace

Diploma project (AITU, SE-2409): "Интеллектуальная система прогнозирования медицинских показателей на основе нейронных сетей". Monorepo github.com/bsanat/Medpredicts (branch `main`), merged with history from the four github.com/MedPredicts repos. Originals backed up in git-ignored `.split-backup/`. CI: `.github/workflows/{ml-ai,backend,frontend}.yml`.

- `documentation/` — proposal, requirements (06), architecture (07), evaluation plan (09), risks (10), supervisor review (11). Russian.
- `frontend/` — React + TypeScript + Vite. Rules: `frontend/AGENTS.md` (skill `medpredict-frontend`).
- `backend/` — Java 17, Spring Boot 3.5. Rules: `backend/SKILL.md`, `backend/AGENTS.md` (skill `java-springboot`). Providers: unavailable (default), `demo`, and `OnnxPredictionModel` (`-Dspring-boot.run.profiles=onnx`, `MEDPREDICT_MODEL_DIRECTORY=../ml-ai/artifacts`). Maven in this sandbox needs `MAVEN_OPTS=-Djava.net.preferIPv4Stack=true`.
- `ml-ai/` — M2 pipeline (`src/medpredict_ml`, `.venv` with Python 3.12), artifacts and reports, ported from `~/MedPredict/ml`. CDC download needs `SSL_CERT_FILE=/etc/ssl/cert.pem`. Rules: `ml-ai/AGENTS.md`, skills `medpredict-ml-engineering`, `medpredict-ml-research`, `scikit-learn`, `scientific-critical-thinking`.

Team: Баймен Али (ML), Сеилхан Оралхан (backend), Бөгенбаев Санат (frontend). Supervisor: Турарова М. К.

AI-analysis roadmap: `AI_ANALYSIS_PLAN.md`. Assistant: backend feature `assistant` (Gemini via `GEMINI_API_KEY` from env or git-ignored `backend/.env`, see `backend/.env.example`; never commit the key), knowledge base `backend/src/main/resources/assistant/knowledge-base.json`, eval protocol `assistant/PROTOCOL.md`, runner `assistant/eval/run_eval.py`. zsh gotcha: write `${m}:generateContent`, not `$m:generate...` (zsh modifiers).

Always keep: research-prototype disclaimers, no diagnosis/treatment advice, no "±MAE" as individual interval, no medical values in logs/storage, contract changes = code + tests + docs together.
