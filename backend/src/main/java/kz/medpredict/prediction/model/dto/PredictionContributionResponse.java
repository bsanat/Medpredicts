package kz.medpredict.prediction.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@AllArgsConstructor
@Schema(description = "Contribution of one input to the estimate relative to the reference participant (model behaviour, not a causal effect).")
public final class PredictionContributionResponse {
    @Schema(description = "Input field key.", example = "age")
    private final String key;
    @Schema(description = "Signed contribution in the result unit.", example = "1.4")
    private final double value;
}
