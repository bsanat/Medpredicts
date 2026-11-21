package kz.medpredict.assistant.infrastructure.llm.impl;

import kz.medpredict.assistant.infrastructure.llm.AssistantLanguageModel;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelReply;
import kz.medpredict.assistant.infrastructure.llm.AssistantModelRequest;
import kz.medpredict.core.exception.AssistantUnavailableException;

public final class UnavailableAssistantLanguageModel implements AssistantLanguageModel {
    @Override
    public boolean available() {
        return false;
    }

    @Override
    public AssistantModelReply generate(AssistantModelRequest request) {
        throw new AssistantUnavailableException();
    }
}
