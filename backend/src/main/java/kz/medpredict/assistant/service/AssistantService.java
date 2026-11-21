package kz.medpredict.assistant.service;

import kz.medpredict.assistant.model.dto.AssistantMessageCreateRequest;
import kz.medpredict.assistant.model.dto.AssistantMessageResponse;
import kz.medpredict.assistant.model.dto.AssistantStatusResponse;

public interface AssistantService {
    AssistantStatusResponse status();

    AssistantMessageResponse reply(AssistantMessageCreateRequest request);
}
