package kz.medpredict.prediction.model.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@AllArgsConstructor
@Schema(description = "Split-conformal interval around the estimate. Coverage is marginal over an external research cohort, "
        + "not a personal guarantee; it is poor when the true value is high.")
public final class PredictionIntervalResponse {
    @Schema(description = "Lower bound in the result unit.", example = "99.8")
    private final double lower;
    @Schema(description = "Upper bound in the result unit.", example = "150.8")
    private final double upper;
    @Schema(description = "Nominal coverage level of the calibration.", example = "0.9")
    private final double level;
    @Schema(description = "Coverage actually measured on the external validation cohort.", example = "0.913")
    private final double measuredCoverage;
    @Schema(description = "Calibration method identifier.", example = "split_symmetric_absolute_residual")
    private final String method;
}
