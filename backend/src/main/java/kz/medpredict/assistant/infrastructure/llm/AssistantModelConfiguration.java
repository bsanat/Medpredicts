package kz.medpredict.assistant.infrastructure.llm;

import com.fasterxml.jackson.databind.ObjectMapper;
import kz.medpredict.assistant.infrastructure.llm.impl.GeminiAssistantLanguageModel;
import kz.medpredict.assistant.infrastructure.llm.impl.UnavailableAssistantLanguageModel;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Gemini is used only when a key is configured; otherwise the assistant reports 503 and never fakes answers. */
@Configuration
@EnableConfigurationProperties(AssistantProperties.class)
public class AssistantModelConfiguration {
    @Bean
    public AssistantLanguageModel assistantLanguageModel(AssistantProperties properties, ObjectMapper objectMapper) {
        if (properties.gemini().apiKey().isEmpty()) {
            return new UnavailableAssistantLanguageModel();
        }
        return new GeminiAssistantLanguageModel(properties.gemini(), objectMapper);
    }
}
