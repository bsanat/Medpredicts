package kz.medpredict.assistant.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@AllArgsConstructor
@Schema(description = "Approved knowledge entry the answer is based on.")
public final class AssistantSourceResponse {
    private final String id;
    private final String title;
    @Schema(description = "Project document the entry was taken from.", example = "ml-ai/docs/model-card.md")
    private final String document;
}
