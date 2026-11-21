package kz.medpredict.prediction.infrastructure.model;

import com.fasterxml.jackson.databind.JsonNode;
import kz.medpredict.prediction.model.dto.PredictionIntervalResponse;

/** Optional conformal calibration shipped in the bundle metadata; offsets are added to the model output. */
public record ModelIntervalCalibration(String method, double level, double lowerOffset, double upperOffset,
        double externalCoverage) {

    static ModelIntervalCalibration read(JsonNode root) {
        var node = root.path("predictionInterval");
        if (node.isMissingNode() || node.isNull()) {
            return null;
        }
        require(node.isObject() && node.path("method").isTextual()
                && node.path("method").textValue().matches("[a-z][a-z0-9_]{0,63}"));
        double level = number(node, "level");
        double lower = number(node, "lowerOffset");
        double upper = number(node, "upperOffset");
        double coverage = number(node, "externalCoverage");
        require(level > 0 && level < 1 && lower <= 0 && upper >= 0 && coverage >= 0 && coverage <= 1);
        return new ModelIntervalCalibration(node.path("method").textValue(), level, lower, upper, coverage);
    }

    public PredictionIntervalResponse around(double estimate) {
        return new PredictionIntervalResponse(estimate + lowerOffset, estimate + upperOffset, level, externalCoverage, method);
    }

    private static double number(JsonNode node, String key) {
        require(node.path(key).isNumber() && Double.isFinite(node.path(key).doubleValue()));
        return node.path(key).doubleValue();
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw new IllegalArgumentException("Incompatible model artifact metadata");
        }
    }
}
