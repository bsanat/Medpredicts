package kz.medpredict.assistant.infrastructure.llm.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import kz.medpredict.assistant.infrastructure.llm.AssistantLanguageModel;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelReply;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelRequest;
import kz.medpredict.assistant.infrastructure.llm.AssistantProperties;
import kz.medpredict.core.exception.AssistantUnavailableException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Google Gemini generateContent client with a JSON response schema. Models are tried in configured order; overload,
 * rate-limit, server and timeout failures move to the next model. Prompts, replies and the key are never logged.
 */
public final class GeminiAssistantLanguageModel implements AssistantLanguageModel {
    private static final Logger LOG = LoggerFactory.getLogger(GeminiAssistantLanguageModel.class);
    private final AssistantProperties.Gemini settings;
    private final ObjectMapper objectMapper;
    private final HttpClient client;

    public GeminiAssistantLanguageModel(AssistantProperties.Gemini settings, ObjectMapper objectMapper) {
        this.settings = settings;
        this.objectMapper = objectMapper;
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5))
                .followRedirects(HttpClient.Redirect.NEVER).build();
    }

    @Override
    public boolean available() {
        return true;
    }

    @Override
    public AssistantModelReply generate(AssistantModelRequest request) {
        byte[] body = requestBody(request);
        for (String model : settings.models()) {
            try {
                var response = client.send(HttpRequest.newBuilder(endpoint(model))
                                .timeout(Duration.ofSeconds(settings.timeoutSeconds()))
                                .header("Content-Type", "application/json")
                                .header("x-goog-api-key", settings.apiKey())
                                .POST(HttpRequest.BodyPublishers.ofByteArray(body)).build(),
                        HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                int status = response.statusCode();
                if (status == 200) {
                    return parse(objectMapper.readTree(response.body()), model);
                }
                LOG.warn("Gemini model {} returned HTTP {}", model, status);
                if (status != 429 && status < 500) {
                    break;  // configuration or request error: other models will fail the same way
                }
            } catch (HttpTimeoutException exception) {
                LOG.warn("Gemini model {} timed out", model);
            } catch (IOException | IllegalArgumentException exception) {
                LOG.warn("Gemini model {} failed: {}", model, exception.getClass().getSimpleName());
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        throw new AssistantUnavailableException();
    }

    private URI endpoint(String model) {
        String base = settings.baseUrl().toString().replaceAll("/+$", "");
        return URI.create(base + "/v1beta/models/" + model + ":generateContent");
    }

    private byte[] requestBody(AssistantModelRequest request) {
        ObjectNode root = objectMapper.createObjectNode();
        root.putObject("systemInstruction").putArray("parts").addObject().put("text", request.systemInstruction());
        ArrayNode contents = root.putArray("contents");
        for (var turn : request.turns()) {
            var content = contents.addObject().put("role", turn.fromUser() ? "user" : "model");
            content.putArray("parts").addObject().put("text", turn.text());
        }
        ObjectNode config = root.putObject("generationConfig");
        config.put("temperature", 0).put("maxOutputTokens", settings.maxOutputTokens()).put("responseMimeType", "application/json");
        if (!settings.thinkingLevel().isEmpty()) {
            // Internal reasoning shares the output budget; keep it low so the JSON answer is not truncated.
            config.putObject("thinkingConfig").put("thinkingLevel", settings.thinkingLevel());
        }
        ObjectNode schema = config.putObject("responseSchema").put("type", "OBJECT");
        ObjectNode properties = schema.putObject("properties");
        properties.putObject("answer").put("type", "STRING");
        properties.putObject("sources").put("type", "ARRAY").putObject("items").put("type", "STRING");
        properties.putObject("refused").put("type", "BOOLEAN");
        schema.putArray("required").add("answer").add("sources").add("refused");
        try {
            return objectMapper.writeValueAsBytes(root);
        } catch (IOException exception) {
            throw new IllegalStateException("Cannot serialize Gemini request", exception);
        }
    }

    private AssistantModelReply parse(JsonNode root, String model) throws IOException {
        var candidate = root.path("candidates").path(0);
        String finish = candidate.path("finishReason").asText("");
        if ("SAFETY".equals(finish) || "PROHIBITED_CONTENT".equals(finish) || root.path("promptFeedback").has("blockReason")) {
            return new AssistantModelReply("", List.of(), true, model);
        }
        if ("MAX_TOKENS".equals(finish)) {
            throw new IllegalArgumentException("Gemini reply truncated");
        }
        StringBuilder text = new StringBuilder();
        for (JsonNode part : candidate.path("content").path("parts")) {
            if (!part.path("thought").asBoolean(false) && part.path("text").isTextual()) {
                text.append(part.path("text").textValue());
            }
        }
        JsonNode reply = objectMapper.readTree(text.toString());
        if (reply == null || !reply.path("answer").isTextual() || !reply.path("refused").isBoolean()
                || !reply.path("sources").isArray()) {
            throw new IllegalArgumentException("Unexpected Gemini reply shape");
        }
        List<String> sources = new ArrayList<>();
        reply.path("sources").forEach(source -> {
            if (source.isTextual()) {
                sources.add(source.textValue());
            }
        });
        String version = root.path("modelVersion").asText(model);
        return new AssistantModelReply(reply.path("answer").textValue(), sources, reply.path("refused").booleanValue(),
                version.isBlank() ? model : version);
    }
}
