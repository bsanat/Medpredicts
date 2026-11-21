package kz.medpredict.assistant.infrastructure.llm;

import java.util.List;

/** Provider-neutral prompt: a system instruction and alternating user/assistant turns ending with the user. */
public record AssistantModelRequest(String systemInstruction, List<Turn> turns) {
    public AssistantModelRequest {
        turns = List.copyOf(turns);
    }

    public record Turn(boolean fromUser, String text) {
    }
}
