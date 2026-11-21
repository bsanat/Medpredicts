package kz.medpredict.assistant.infrastructure.llm;

import java.net.URI;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "medpredict.assistant")
public record AssistantProperties(Gemini gemini, int requestsPerMinute) {
    public AssistantProperties {
        if (gemini == null) {
            throw new IllegalArgumentException("Assistant Gemini settings are required.");
        }
        if (requestsPerMinute < 1 || requestsPerMinute > 600) {
            throw new IllegalArgumentException("Assistant requests per minute must be 1–600.");
        }
    }

    public record Gemini(String apiKey, List<String> models, URI baseUrl, int timeoutSeconds, int maxOutputTokens,
            String thinkingLevel) {
        public Gemini {
            apiKey = apiKey == null ? "" : apiKey.trim();
            thinkingLevel = thinkingLevel == null ? "" : thinkingLevel.trim();
            if (!thinkingLevel.matches("|minimal|low|medium|high")) {
                throw new IllegalArgumentException("Gemini thinking level must be blank, minimal, low, medium or high.");
            }
            models = models == null ? List.of() : models.stream().map(String::trim).filter(m -> !m.isEmpty()).toList();
            if (models.isEmpty() || models.stream().anyMatch(m -> !m.matches("[a-z0-9][a-z0-9.-]{0,63}"))) {
                throw new IllegalArgumentException("At least one valid Gemini model name is required.");
            }
            if (baseUrl == null || !"https".equals(baseUrl.getScheme()) && !"http".equals(baseUrl.getScheme())) {
                throw new IllegalArgumentException("Gemini base URL must be HTTP(S).");
            }
            if (timeoutSeconds < 1 || timeoutSeconds > 120 || maxOutputTokens < 64 || maxOutputTokens > 8192) {
                throw new IllegalArgumentException("Invalid Gemini timeout or output limit.");
            }
        }

        @Override
        public String toString() {
            return "Gemini[models=" + models + ", baseUrl=" + baseUrl + ", apiKey=" + (apiKey.isEmpty() ? "absent" : "present") + "]";
        }
    }
}
