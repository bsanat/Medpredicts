package kz.medpredict.assistant.infrastructure.llm.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelRequest;
import kz.medpredict.assistant.infrastructure.llm.AssistantProperties;
import kz.medpredict.core.exception.AssistantUnavailableException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class GeminiAssistantLanguageModelTest {
    private final ObjectMapper mapper = new ObjectMapper();
    private final Map<String, Integer> statusByModel = new ConcurrentHashMap<>();
    private final Map<String, String> bodyByModel = new ConcurrentHashMap<>();
    private final List<String> calls = new ArrayList<>();
    private final List<String> keys = new ArrayList<>();
    private final List<String> requestBodies = new ArrayList<>();
    private HttpServer server;

    @BeforeEach
    void start() throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1beta/models/", exchange -> {
            String model = exchange.getRequestURI().getPath().replace("/v1beta/models/", "").replace(":generateContent", "");
            synchronized (calls) {
                calls.add(model);
                keys.add(exchange.getRequestHeaders().getFirst("x-goog-api-key"));
                requestBodies.add(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            }
            if (model.equals("slow")) {
                try {
                    Thread.sleep(2500);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                }
            }
            byte[] body = bodyByModel.getOrDefault(model, "{}").getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(statusByModel.getOrDefault(model, 200), body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool());
        server.start();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    private GeminiAssistantLanguageModel model(String... models) {
        var settings = new AssistantProperties.Gemini("test-key", List.of(models),
                URI.create("http://127.0.0.1:" + server.getAddress().getPort()), 1, 512, "low");
        return new GeminiAssistantLanguageModel(settings, mapper);
    }

    private static String reply(String json, String finish) throws Exception {
        var root = new ObjectMapper().createObjectNode();
        var candidate = root.putArray("candidates").addObject().put("finishReason", finish);
        var parts = candidate.putObject("content").putArray("parts");
        parts.addObject().put("text", "internal reasoning").put("thought", true);
        parts.addObject().put("text", json);
        root.put("modelVersion", "gemini-test-001");
        return root.toString();
    }

    private static AssistantModelRequest request() {
        return new AssistantModelRequest("RULES", List.of(new AssistantModelRequest.Turn(true, "LANGUAGE: ru\nQUESTION: ?")));
    }

    @Test
    void parsesStructuredReplySkippingThoughtsAndSendsKeyAndSchema() throws Exception {
        bodyByModel.put("primary", reply("{\"answer\":\"Ответ\",\"sources\":[\"interval\"],\"refused\":false}", "STOP"));
        var result = model("primary").generate(request());
        assertThat(result.answer()).isEqualTo("Ответ");
        assertThat(result.sources()).containsExactly("interval");
        assertThat(result.refused()).isFalse();
        assertThat(result.model()).isEqualTo("gemini-test-001");
        assertThat(keys).containsExactly("test-key");
        var sent = mapper.readTree(requestBodies.get(0));
        assertThat(sent.path("systemInstruction").path("parts").get(0).path("text").textValue()).isEqualTo("RULES");
        assertThat(sent.path("contents").get(0).path("role").textValue()).isEqualTo("user");
        assertThat(sent.path("generationConfig").path("responseMimeType").textValue()).isEqualTo("application/json");
        assertThat(sent.path("generationConfig").path("responseSchema").path("required").toString()).contains("answer", "sources", "refused");
        assertThat(sent.path("generationConfig").path("temperature").intValue()).isZero();
        assertThat(sent.path("generationConfig").path("thinkingConfig").path("thinkingLevel").textValue()).isEqualTo("low");
    }

    @Test
    void overloadedAndTimedOutModelsFallBackToTheNextModel() throws Exception {
        statusByModel.put("busy", 503);
        bodyByModel.put("backup", reply("{\"answer\":\"ok\",\"sources\":[],\"refused\":true}", "STOP"));
        assertThat(model("busy", "slow", "backup").generate(request()).refused()).isTrue();
        assertThat(calls).containsExactly("busy", "slow", "backup");
    }

    @Test
    void clientErrorsStopAndAllFailuresAreUnavailable() throws Exception {
        statusByModel.put("forbidden", 403);
        assertThatThrownBy(() -> model("forbidden", "never").generate(request())).isInstanceOf(AssistantUnavailableException.class);
        assertThat(calls).containsExactly("forbidden");
        bodyByModel.put("garbled", reply("not json", "STOP"));
        assertThatThrownBy(() -> model("garbled").generate(request())).isInstanceOf(AssistantUnavailableException.class);
        bodyByModel.put("truncated", reply("{\"answer\":\"ok\",\"sources\":[],\"refused\":false}", "MAX_TOKENS"));
        assertThatThrownBy(() -> model("truncated").generate(request())).isInstanceOf(AssistantUnavailableException.class);
        bodyByModel.put("shape", reply("{\"answer\":1,\"sources\":[],\"refused\":false}", "STOP"));
        assertThatThrownBy(() -> model("shape").generate(request())).isInstanceOf(AssistantUnavailableException.class);
    }

    @Test
    void safetyBlockedReplyBecomesEmptyRefusal() throws Exception {
        bodyByModel.put("safe", reply("", "SAFETY"));
        var result = model("safe").generate(request());
        assertThat(result.refused()).isTrue();
        assertThat(result.answer()).isEmpty();
    }

    @Test
    void settingsNeverPrintTheKey() {
        var settings = new AssistantProperties.Gemini("secret-value", List.of("m"), URI.create("https://example.test"), 5, 512, "");
        assertThat(settings.toString()).doesNotContain("secret-value").contains("present");
        assertThatThrownBy(() -> new AssistantProperties.Gemini("", List.of("Bad Model"), URI.create("https://x.test"), 5, 512, ""))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
