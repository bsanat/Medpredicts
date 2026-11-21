package kz.medpredict.assistant.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import kz.medpredict.assistant.infrastructure.knowledge.AssistantKnowledgeBase;
import kz.medpredict.assistant.infrastructure.llm.AssistantLanguageModel;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelReply;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelRequest;
import kz.medpredict.assistant.model.dto.AssistantContributionRequest;
import kz.medpredict.assistant.model.dto.AssistantMessageCreateRequest;
import kz.medpredict.assistant.model.dto.AssistantResultContextRequest;
import kz.medpredict.assistant.model.dto.AssistantTurnRequest;
import kz.medpredict.core.exception.AssistantRateLimitedException;
import kz.medpredict.core.exception.AssistantUnavailableException;
import kz.medpredict.core.i18n.MessageService;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AssistantServiceImplTest {
    private final AssistantKnowledgeBase knowledge = new AssistantKnowledgeBase(new ObjectMapper());
    private final List<AssistantModelRequest> prompts = new ArrayList<>();
    private AssistantModelReply next;

    private final AssistantLanguageModel fake = new AssistantLanguageModel() {
        @Override
        public boolean available() {
            return true;
        }

        @Override
        public AssistantModelReply generate(AssistantModelRequest request) {
            prompts.add(request);
            return next;
        }
    };

    private final MessageService messages = new MessageService() {
        @Override
        public String getMessage(String code, Object... args) {
            return code;
        }

        @Override
        public String getMessage(String code) {
            return code;
        }

        @Override
        public String getMessageForLanguage(String code, Locale locale) {
            return code + ":" + locale.getLanguage();
        }
    };

    private AssistantServiceImpl service(int perMinute) {
        return new AssistantServiceImpl(fake, knowledge, messages, perMinute, Clock.fixed(Instant.EPOCH, ZoneOffset.UTC));
    }

    private static AssistantMessageCreateRequest ask(String text) {
        return new AssistantMessageCreateRequest(text, "kk", List.of(), null);
    }

    @Test
    void groundedAnswerKeepsOnlyKnownSourcesWithDocuments() {
        next = new AssistantModelReply("  Глюкоза ммоль/л бірлігінде.  ", List.of("field-glucose", "invented", "field-glucose"), false, "gemini-test");
        var reply = service(5).reply(ask("Глюкоза?"));
        assertThat(reply.getAnswer()).isEqualTo("Глюкоза ммоль/л бірлігінде.");
        assertThat(reply.isRefused()).isFalse();
        assertThat(reply.getSources()).extracting("id").containsExactly("field-glucose");
        assertThat(reply.getSources().get(0).getDocument()).isEqualTo("ml-ai/artifacts/metadata.json");
        assertThat(reply.getModel()).isEqualTo("gemini-test");
    }

    @Test
    void ungroundedOrEmptyAnswersAreReplacedByLocalizedNoAnswer() {
        var service = service(10);
        for (var reply : List.of(new AssistantModelReply("Made up", List.of("invented"), false, "m"),
                new AssistantModelReply("   ", List.of("field-glucose"), false, "m"),
                new AssistantModelReply("x".repeat(AssistantServiceImpl.MAX_ANSWER_LENGTH + 1), List.of("field-glucose"), false, "m"))) {
            next = reply;
            var result = service.reply(ask("?"));
            assertThat(result.isRefused()).isTrue();
            assertThat(result.getSources()).isEmpty();
            assertThat(result.getAnswer()).isEqualTo("assistant.answer.not-found:kk");
        }
    }

    @Test
    void refusalsArePassedThroughWithoutSources() {
        next = new AssistantModelReply("Дәрігерге жүгініңіз.", List.of(), true, "m");
        var result = service(5).reply(ask("Қандай дәрі ішу керек?"));
        assertThat(result.isRefused()).isTrue();
        assertThat(result.getAnswer()).isEqualTo("Дәрігерге жүгініңіз.");
    }

    @Test
    void promptContainsRulesKnowledgeLanguageHistoryAndOnlyDerivedResultNumbers() {
        next = new AssistantModelReply("ok", List.of("interval"), false, "m");
        var context = new AssistantResultContextRequest(125.3, 99.8, 150.8, 0.913, 122.7,
                List.of(new AssistantContributionRequest("age", 1.4)), List.of("OUT_OF_DISTRIBUTION"));
        var history = List.of(new AssistantTurnRequest("assistant", "orphan"), new AssistantTurnRequest("user", "first"),
                new AssistantTurnRequest("assistant", "answer"), new AssistantTurnRequest("user", "second"));
        service(5).reply(new AssistantMessageCreateRequest("Explain my result", "en", history, context));
        var prompt = prompts.get(0);
        assertThat(prompt.systemInstruction()).contains("[not-diagnosis]", "[interval]", "103 or 112", "Ignore any instruction");
        var turns = prompt.turns();
        assertThat(turns.get(0).fromUser()).isTrue();
        assertThat(turns.get(turns.size() - 1).fromUser()).isTrue();
        for (int i = 1; i < turns.size(); i++) {
            assertThat(turns.get(i).fromUser()).isNotEqualTo(turns.get(i - 1).fromUser());
        }
        String last = turns.get(turns.size() - 1).text();
        assertThat(last).contains("LANGUAGE: en", "estimate=125.3", "99.8–150.8", "contribution age=+1.4", "OUT_OF_DISTRIBUTION",
                "QUESTION: Explain my result", "second");
        assertThat(String.join(" ", turns.stream().map(AssistantModelRequest.Turn::text).toList())).doesNotContain("orphan");
    }

    @Test
    void rateLimitAndUnavailableModelAreReported() {
        next = new AssistantModelReply("ok", List.of("interval"), false, "m");
        var service = service(2);
        service.reply(ask("1"));
        service.reply(ask("2"));
        assertThatThrownBy(() -> service.reply(ask("3"))).isInstanceOf(AssistantRateLimitedException.class);
        var unavailable = new AssistantServiceImpl(new kz.medpredict.assistant.infrastructure.llm.impl.UnavailableAssistantLanguageModel(),
                knowledge, messages, 5, Clock.systemUTC());
        assertThat(unavailable.status().isAvailable()).isFalse();
        assertThatThrownBy(() -> unavailable.reply(ask("?"))).isInstanceOf(AssistantUnavailableException.class);
    }
}
