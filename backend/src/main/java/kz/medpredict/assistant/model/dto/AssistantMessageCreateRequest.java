package kz.medpredict.assistant.model.dto;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.util.List;
import kz.medpredict.core.i18n.MessageCode;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@Schema(description = "A question for the documentation assistant. Do not include personal data.")
public final class AssistantMessageCreateRequest {
    @NotBlank(message = MessageCode.ASSISTANT_MESSAGE_LENGTH)
    @Size(max = 1000, message = MessageCode.ASSISTANT_MESSAGE_LENGTH)
    @Schema(example = "Что означает интервал 90%?")
    private final String message;
    @NotBlank(message = MessageCode.ASSISTANT_LANGUAGE)
    @Pattern(regexp = "ru|kk|en", message = MessageCode.ASSISTANT_LANGUAGE)
    @Schema(allowableValues = {"ru", "kk", "en"})
    private final String language;
    @Valid
    @Size(max = 6, message = MessageCode.INVALID_REQUEST)
    private final List<AssistantTurnRequest> history;
    @Valid
    private final AssistantResultContextRequest resultContext;

    public AssistantMessageCreateRequest(String message, String language, List<AssistantTurnRequest> history,
            AssistantResultContextRequest resultContext) {
        this.message = message;
        this.language = language;
        this.history = history == null ? List.of() : List.copyOf(history);
        this.resultContext = resultContext;
    }

    /** Unknown keys (for example raw medical inputs) are rejected instead of silently ignored. */
    @JsonAnySetter
    void rejectUnknown(String name, Object value) {
        throw new IllegalArgumentException("Unknown field");
    }
}
