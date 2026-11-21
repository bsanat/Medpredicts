package kz.medpredict.assistant.infrastructure.llm;

/** External language model behind the documentation assistant. Implementations never log prompts or replies. */
public interface AssistantLanguageModel {
    boolean available();

    /** @throws kz.medpredict.core.exception.AssistantUnavailableException when no configured model can answer */
    AssistantModelReply generate(AssistantModelRequest request);
}
