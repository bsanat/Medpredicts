package kz.medpredict.assistant.model.dto;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import kz.medpredict.core.i18n.MessageCode;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@AllArgsConstructor
@Schema(description = "One earlier turn of the conversation, kept only for this request.")
public final class AssistantTurnRequest {
    @NotBlank(message = MessageCode.FIELD_REQUIRED)
    @Pattern(regexp = "user|assistant", message = MessageCode.INVALID_REQUEST)
    @Schema(allowableValues = {"user", "assistant"})
    private final String role;
    @NotBlank(message = MessageCode.FIELD_REQUIRED)
    @Size(max = 2000, message = MessageCode.ASSISTANT_MESSAGE_LENGTH)
    private final String text;

    /** Unknown keys (for example raw medical inputs) are rejected instead of silently ignored. */
    @JsonAnySetter
    void rejectUnknown(String name, Object value) {
        throw new IllegalArgumentException("Unknown field");
    }
}
