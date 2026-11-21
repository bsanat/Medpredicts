package kz.medpredict.prediction.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.List;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@Schema(description = "Exact baseline Shapley decomposition: referenceValue plus all contributions equals predictedValue. "
        + "Describes model behaviour relative to the average training participant, not causes.")
public final class PredictionExplanationResponse {
    @Schema(description = "Model output for the reference participant, in the result unit.", example = "122.7")
    private final double referenceValue;
    @Schema(description = "One contribution per input, in metadata field order.")
    private final List<PredictionContributionResponse> contributions;
    @Schema(description = "Explanation method identifier.", example = "exact_baseline_shapley")
    private final String method;

    public PredictionExplanationResponse(double referenceValue, List<PredictionContributionResponse> contributions, String method) {
        this.referenceValue = referenceValue;
        this.contributions = List.copyOf(contributions);
        this.method = method;
    }
}
