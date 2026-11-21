package kz.medpredict.assistant.service.impl;

import java.time.Clock;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import kz.medpredict.assistant.infrastructure.knowledge.AssistantKnowledgeBase;
import kz.medpredict.assistant.infrastructure.llm.AssistantLanguageModel;
import kz.medpredict.assistant.infrastructure.llm.AssistantProperties;
import kz.medpredict.assistant.model.dto.AssistantMessageCreateRequest;
import kz.medpredict.assistant.model.dto.AssistantMessageResponse;
import kz.medpredict.assistant.model.dto.AssistantSourceResponse;
import kz.medpredict.assistant.model.dto.AssistantStatusResponse;
import kz.medpredict.assistant.service.AssistantService;
import kz.medpredict.core.exception.AssistantRateLimitedException;
import kz.medpredict.core.exception.AssistantUnavailableException;
import kz.medpredict.core.exception.InvalidRequestException;
import kz.medpredict.core.i18n.MessageCode;
import kz.medpredict.core.i18n.MessageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AssistantServiceImpl implements AssistantService {
    static final int MAX_ANSWER_LENGTH = 2000;
    private final AssistantLanguageModel model;
    private final AssistantKnowledgeBase knowledge;
    private final MessageService messages;
    private final int requestsPerMinute;
    private final Clock clock;
    private final Deque<Long> recent = new ArrayDeque<>();

    @Autowired
    public AssistantServiceImpl(AssistantLanguageModel model, AssistantKnowledgeBase knowledge, MessageService messages,
            AssistantProperties properties) {
        this(model, knowledge, messages, properties.requestsPerMinute(), Clock.systemUTC());
    }

    AssistantServiceImpl(AssistantLanguageModel model, AssistantKnowledgeBase knowledge, MessageService messages,
            int requestsPerMinute, Clock clock) {
        this.model = model;
        this.knowledge = knowledge;
        this.messages = messages;
        this.requestsPerMinute = requestsPerMinute;
        this.clock = clock;
    }

    @Override
    public AssistantStatusResponse status() {
        return new AssistantStatusResponse(model.available(), knowledge.version());
    }

    @Override
    public AssistantMessageResponse reply(AssistantMessageCreateRequest request) {
        if (request == null) {
            throw new InvalidRequestException();
        }
        if (!model.available()) {
            throw new AssistantUnavailableException();
        }
        acquire();
        var reply = model.generate(AssistantPromptBuilder.build(request, knowledge));
        if (reply == null || reply.answer() == null) {
            throw new AssistantUnavailableException();
        }
        List<AssistantSourceResponse> sources = reply.sources().stream().distinct()
                .map(knowledge::find).flatMap(java.util.Optional::stream)
                .map(entry -> new AssistantSourceResponse(entry.id(), entry.title(), entry.source())).toList();
        String answer = reply.answer().strip();
        boolean grounded = reply.refused() || !sources.isEmpty();
        if (answer.isEmpty() || answer.length() > MAX_ANSWER_LENGTH || !grounded) {
            // Ungrounded or unusable output is never shown; the user gets a localized "no approved answer" message.
            var locale = Locale.forLanguageTag(request.getLanguage());
            return new AssistantMessageResponse(messages.getMessageForLanguage(MessageCode.ASSISTANT_NO_ANSWER, locale), true, List.of(),
                    Objects.requireNonNullElse(reply.model(), ""));
        }
        return new AssistantMessageResponse(answer, reply.refused(), sources, Objects.requireNonNullElse(reply.model(), ""));
    }

    private synchronized void acquire() {
        long now = clock.millis();
        while (!recent.isEmpty() && now - recent.peekFirst() >= 60_000) {
            recent.removeFirst();
        }
        if (recent.size() >= requestsPerMinute) {
            throw new AssistantRateLimitedException();
        }
        recent.addLast(now);
    }
}
