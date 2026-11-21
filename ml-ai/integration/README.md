# Подключение bundle к backend

Bundle **MedPredict-SBP-mlp-v2** находится в [`../artifacts/`](../artifacts/): `model.onnx`, `metadata.json`, `golden.json`. SHA-256 модели: `3a4be6a1746f4230a5dba7346e3948e56cabc400540c91b0fbcd7882299dbe2a`.

Провайдер `OnnxPredictionModel` находится в [MedPredicts/backend](https://github.com/MedPredicts/backend) (ветка `feat/onnx-provider`, до слияния в `main`). Из корня backend:

```sh
export MEDPREDICT_MODEL_DIRECTORY=/absolute/path/to/ml-ai/artifacts
export MEDPREDICT_MODEL_VERSION=MedPredict-SBP-mlp-v2   # необязательный пин версии
sh mvnw test -Dmedpredict.model.test-directory="$MEDPREDICT_MODEL_DIRECTORY"
sh mvnw spring-boot:run -Dspring-boot.run.profiles=onnx
```

## Контракт bundle

- Вход ONNX `features`: float32 `[batch,5]`, порядок `age, sex, bmi, totalCholesterol, glucose`; `male=0`, `female=1`; холестерин и глюкоза плазмы натощак в ммоль/л.
- Выход `predictedValue`: float32 `[batch,1]`, мм рт. ст. Масштабирование и обратное преобразование встроены в граф.
- При старте backend проверяет schema version, SHA-256 загруженных байтов, версию, порядок, единицы, категории и формы тензоров. Ошибка артефакта или inference → 503 `MODEL_UNAVAILABLE`, без подстановки demo.
- `golden.json` содержит 4 синтетических входа и выходы Python; Java сверяет их с допуском 0,01 мм рт. ст.

## Проверка (22.11.2026)

JDK 17.0.9, ONNX Runtime Java 1.24.3, macOS arm64: **48 тестов backend, 0 падений, 0 пропусков**; максимальное расхождение Python/Java **7,63·10⁻⁶ мм рт. ст.** Если Maven не может скачать зависимости (`No route to host` при рабочем curl), помогает `MAVEN_OPTS=-Djava.net.preferIPv4Stack=true`.

Модель — исследовательская оценка текущего САД; не для диагностики и не для изменения лечения. Применимость к населению Казахстана не установлена.
