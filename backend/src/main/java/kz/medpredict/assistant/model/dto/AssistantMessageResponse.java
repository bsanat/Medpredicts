package kz.medpredict.assistant.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@Schema(description = "Assistant reply grounded in approved documentation. It never contains or changes a prediction.")
public final class AssistantMessageResponse {
    private final String answer;
    @Schema(description = "True when the assistant declined (diagnosis, treatment, emergency, off-topic or no approved answer).")
    private final boolean refused;
    private final List<AssistantSourceResponse> sources;
    @Schema(description = "Language model that produced the reply.", example = "gemini-3.5-flash")
    private final String model;

    public AssistantMessageResponse(String answer, boolean refused, List<AssistantSourceResponse> sources, String model) {
        this.answer = answer;
        this.refused = refused;
        this.sources = List.copyOf(sources);
        this.model = model;
    }
}
