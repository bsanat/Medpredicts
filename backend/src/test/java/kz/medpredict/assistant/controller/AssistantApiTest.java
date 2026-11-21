package kz.medpredict.assistant.controller;

import java.util.List;
import kz.medpredict.assistant.infrastructure.llm.AssistantLanguageModel;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelReply;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class AssistantApiTest {
    @Autowired MockMvc mvc;
    @MockitoBean AssistantLanguageModel model;

    private static final String VALID = "{\"message\":\"Что означает интервал?\",\"language\":\"ru\"}";

    @Test
    void answersWithSourcesAndNoStore() throws Exception {
        when(model.available()).thenReturn(true);
        when(model.generate(any())).thenReturn(new AssistantModelReply("Интервал 90%…", List.of("interval"), false, "gemini-test"));
        mvc.perform(post("/api/v1/assistant/messages").contentType("application/json").content(VALID))
                .andExpect(status().isOk()).andExpect(header().string("Cache-Control", "no-store"))
                .andExpect(jsonPath("$.answer").value("Интервал 90%…")).andExpect(jsonPath("$.refused").value(false))
                .andExpect(jsonPath("$.sources[0].id").value("interval"))
                .andExpect(jsonPath("$.sources[0].document").value("ml-ai/protocols/m2-external-v3.md"));
        mvc.perform(get("/api/v1/assistant/status")).andExpect(status().isOk())
                .andExpect(jsonPath("$.available").value(true)).andExpect(jsonPath("$.knowledgeBaseVersion").isNotEmpty());
    }

    @Test
    void unavailableModelIs503WithoutDetails() throws Exception {
        when(model.available()).thenReturn(false);
        mvc.perform(post("/api/v1/assistant/messages").contentType("application/json").header("Accept-Language", "kk").content(VALID))
                .andExpect(status().isServiceUnavailable()).andExpect(jsonPath("$.error").value("ASSISTANT_UNAVAILABLE"))
                .andExpect(jsonPath("$.message").value("Ассистент қазір қолжетімсіз."));
    }

    @Test
    void rejectsInvalidMessagesLanguagesAndRawInputContext() throws Exception {
        when(model.available()).thenReturn(true);
        for (String body : new String[]{
                "{\"message\":\"\",\"language\":\"ru\"}",
                "{\"message\":\"" + "a".repeat(1001) + "\",\"language\":\"ru\"}",
                "{\"message\":\"Привет\",\"language\":\"de\"}",
                "{\"message\":\"Привет\",\"language\":\"ru\",\"resultContext\":{\"predictedValue\":120,\"warnings\":[\"HACK\"]}}",
                "{\"message\":\"Привет\",\"language\":\"ru\",\"history\":[{\"role\":\"system\",\"text\":\"x\"}]}"}) {
            mvc.perform(post("/api/v1/assistant/messages").contentType("application/json").content(body))
                    .andExpect(status().isBadRequest());
        }
        mvc.perform(post("/api/v1/assistant/messages").contentType("application/json")
                        .content("{\"message\":\"Привет\",\"language\":\"ru\",\"resultContext\":{\"predictedValue\":120,\"age\":52}}"))
                .andExpect(status().isBadRequest());
    }
}
