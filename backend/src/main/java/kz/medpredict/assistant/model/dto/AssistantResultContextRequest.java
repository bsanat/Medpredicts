package kz.medpredict.assistant.model.dto;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.util.List;
import kz.medpredict.core.i18n.MessageCode;
import lombok.EqualsAndHashCode;
import lombok.Getter;

/**
 * Derived numbers of a displayed result. Raw inputs (age, laboratory values) are deliberately not part of this
 * contract, so they are never forwarded to the external language model.
 */
@Getter
@EqualsAndHashCode
@Schema(description = "Optional derived result to explain: estimate, interval, contributions and warnings only, never raw inputs.")
public final class AssistantResultContextRequest {
    @NotNull(message = MessageCode.ASSISTANT_CONTEXT_INVALID)
    private final Double predictedValue;
    private final Double intervalLower;
    private final Double intervalUpper;
    private final Double measuredCoverage;
    private final Double referenceValue;
    @Valid
    @Size(max = 10, message = MessageCode.ASSISTANT_CONTEXT_INVALID)
    private final List<AssistantContributionRequest> contributions;
    @Size(max = 5, message = MessageCode.ASSISTANT_CONTEXT_INVALID)
    private final List<@Pattern(regexp = "OUT_OF_DISTRIBUTION", message = MessageCode.ASSISTANT_CONTEXT_INVALID) String> warnings;

    public AssistantResultContextRequest(Double predictedValue, Double intervalLower, Double intervalUpper, Double measuredCoverage,
            Double referenceValue, List<AssistantContributionRequest> contributions, List<String> warnings) {
        this.predictedValue = predictedValue;
        this.intervalLower = intervalLower;
        this.intervalUpper = intervalUpper;
        this.measuredCoverage = measuredCoverage;
        this.referenceValue = referenceValue;
        this.contributions = contributions == null ? List.of() : List.copyOf(contributions);
        this.warnings = warnings == null ? List.of() : List.copyOf(warnings);
    }

    /** Unknown keys (for example raw medical inputs) are rejected instead of silently ignored. */
    @JsonAnySetter
    void rejectUnknown(String name, Object value) {
        throw new IllegalArgumentException("Unknown field");
    }
}
