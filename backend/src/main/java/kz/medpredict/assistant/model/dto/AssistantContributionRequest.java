package kz.medpredict.assistant.model.dto;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import kz.medpredict.core.i18n.MessageCode;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@AllArgsConstructor
public final class AssistantContributionRequest {
    @NotBlank(message = MessageCode.ASSISTANT_CONTEXT_INVALID)
    @Pattern(regexp = "[A-Za-z][A-Za-z0-9]{0,40}", message = MessageCode.ASSISTANT_CONTEXT_INVALID)
    private final String key;
    @NotNull(message = MessageCode.ASSISTANT_CONTEXT_INVALID)
    private final Double value;

    /** Unknown keys (for example raw medical inputs) are rejected instead of silently ignored. */
    @JsonAnySetter
    void rejectUnknown(String name, Object value) {
        throw new IllegalArgumentException("Unknown field");
    }
}
