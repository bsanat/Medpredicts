package kz.medpredict.assistant.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@AllArgsConstructor
@Schema(description = "Whether the assistant is configured; never exposes keys or provider errors.")
public final class AssistantStatusResponse {
    private final boolean available;
    private final String knowledgeBaseVersion;
}
