package kz.medpredict.prediction.model.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import java.util.List;
import lombok.EqualsAndHashCode;
import lombok.Getter;

@Getter
@EqualsAndHashCode
@Schema(description = "A finite prediction with the version actually used and an educational disclaimer.")
public final class PredictionResultResponse {
    @Schema(description = "Finite numeric output; the demo profile always returns 134.2.")
    private final double predictedValue;
    @NotBlank
    @Schema(description = "Unit supplied by the active metadata; not inferred or translated automatically.")
    private final String unit;
    @NotBlank
    @Schema(description = "Version of the provider or artifact; prediction results use the version actually executed.")
    private final String modelVersion;
    @NotBlank
    @Schema(description = "Educational limitations, including explicit labeling for fixed demo responses.")
    private final String disclaimer;
    @Schema(description = "Meaning of the output; estimated_current_measurement does not forecast future pressure.")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private final String predictionType;
    @Schema(description = "True only for an explicitly selected fixed demo provider.")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private final Boolean demo;
    @Schema(description = "Optional calibrated interval; absent for demo and for bundles without an approved calibration.")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private final PredictionIntervalResponse interval;
    @Schema(description = "Optional per-input contributions; absent for demo and bundles without an explanation reference.")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private final PredictionExplanationResponse explanation;
    @Schema(description = "Machine-readable warning codes, e.g. OUT_OF_DISTRIBUTION for an atypical combination of inputs.")
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private final List<String> warnings;

    public PredictionResultResponse(double predictedValue, String unit, String modelVersion, String disclaimer) {
        this(predictedValue, unit, modelVersion, disclaimer, null, null, null);
    }

    public PredictionResultResponse(double predictedValue, String unit, String modelVersion, String disclaimer,
            String predictionType, Boolean demo) {
        this(predictedValue, unit, modelVersion, disclaimer, predictionType, demo, null);
    }

    public PredictionResultResponse(double predictedValue, String unit, String modelVersion, String disclaimer,
            String predictionType, Boolean demo, PredictionIntervalResponse interval) {
        this(predictedValue, unit, modelVersion, disclaimer, predictionType, demo, interval, null, List.of());
    }

    public PredictionResultResponse(double predictedValue, String unit, String modelVersion, String disclaimer,
            String predictionType, Boolean demo, PredictionIntervalResponse interval,
            PredictionExplanationResponse explanation, List<String> warnings) {
        this.predictedValue = predictedValue;
        this.unit = unit;
        this.modelVersion = modelVersion;
        this.disclaimer = disclaimer;
        this.predictionType = predictionType;
        this.demo = demo;
        this.interval = interval;
        this.explanation = explanation;
        this.warnings = warnings == null ? List.of() : List.copyOf(warnings);
    }
}
