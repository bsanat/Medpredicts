# MedPredict

Интеллектуальная система прогнозирования медицинских показателей на основе нейронных сетей. Дипломный проект Astana IT University, Software Engineering, SE-2409.

**Исследовательский прототип:** не предназначен для диагностики, лечения или изменения лекарств, не заменяет тонометр, клиническая валидация для Казахстана не проводилась.

## Структура монорепозитория

| Каталог | Содержимое | Ответственный |
|---|---|---|
| [`documentation/`](documentation/) | Предложение, интервью, обзор литературы, требования, архитектура, UX, план оценки, риски, отзывы руководителя | Команда |
| [`ml-ai/`](ml-ai/) | Воспроизводимый pipeline M2 (NHANES → MLP/baselines → ONNX), отчёты, model card, bundle `MedPredict-SBP-mlp-v2` | Баймен Али |
| [`backend/`](backend/) | Java 17 / Spring Boot 3.5 REST API, валидация, i18n, ONNX Runtime provider | Сеилхан Оралхан |
| [`frontend/`](frontend/) | React + TypeScript + Vite, форма, RU/KK/EN, результат, PDF | Бөгенбаев Санат |
| [`assistant/`](assistant/) | Протокол и набор вопросов для оценки AI-ассистента (Google Gemini), результаты прогонов | Команда |
| [`AI_ANALYSIS_PLAN.md`](AI_ANALYSIS_PLAN.md) | План усиления ИИ-анализа (фазы 0–7) | — |

Научный руководитель — Турарова Маржан Кабдыкалиевна. История коммитов исходных репозиториев [MedPredicts](https://github.com/MedPredicts) сохранена.

## Быстрый запуск (M2 end-to-end)

```sh
# 1. ML: тесты (bundle уже лежит в ml-ai/artifacts)
cd ml-ai
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest
cd ..

# 2. Backend с обученной моделью (порт 8080)
cd backend
sh mvnw test -Dmedpredict.model.test-directory="$PWD/../ml-ai/artifacts"
MEDPREDICT_MODEL_DIRECTORY="$PWD/../ml-ai/artifacts" sh mvnw spring-boot:run -Dspring-boot.run.profiles=onnx
```

```sh
# 3. Frontend (в другом терминале; Vite proxy → 127.0.0.1:8080)
cd frontend
npm ci
npm run dev
```

Откройте http://127.0.0.1:5173/#/predict. Для AI-ассистента (`#/assistant`) скопируйте `backend/.env.example` в `backend/.env` и впишите `GEMINI_API_KEY` (файл не попадает в git; без ключа ассистент отвечает 503). Подробности — в README каждого каталога. Полное воспроизведение обучения из данных CDC: [`ml-ai/README.md`](ml-ai/README.md#установка-запуск-и-тестирование).

## Текущее состояние M2

Оценка текущего систолического давления по возрасту, полу, ИМТ, общему холестерину и глюкозе плазмы натощак (NHANES 2017–March 2020, 3 318 участников). Weighted test MAE: MLP 11,225, Ridge 11,268 мм рт. ст. — превосходство нейросети не установлено. При фактическом САД ≥150 все 42 тестовых случая занижены (MAE 41,366). Python↔Java ONNX parity ≤ 7,6·10⁻⁶ мм рт. ст.

## CI

GitHub Actions запускает проверки только для изменённой части: [`ml-ai`](.github/workflows/ml-ai.yml) (pytest + SHA-256 bundle), [`backend`](.github/workflows/backend.yml) (Maven tests + parity на bundle из `ml-ai/artifacts`), [`frontend`](.github/workflows/frontend.yml) (typecheck, lint, vitest, build).
