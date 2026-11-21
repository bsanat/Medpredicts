package kz.medpredict.assistant.infrastructure.llm;

import java.util.List;

/** Structured reply parsed from the provider; validated again by the service before it reaches a client. */
public record AssistantModelReply(String answer, List<String> sources, boolean refused, String model) {
    public AssistantModelReply {
        sources = sources == null ? List.of() : List.copyOf(sources);
    }
}
