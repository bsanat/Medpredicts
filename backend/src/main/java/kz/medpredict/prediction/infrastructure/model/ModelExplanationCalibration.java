package kz.medpredict.prediction.infrastructure.model;

import com.fasterxml.jackson.databind.JsonNode;

/** Reference point for exact baseline Shapley values: absent features take these values. */
public record ModelExplanationCalibration(String method, double[] reference) {
    public static final String METHOD = "exact_baseline_shapley";

    public ModelExplanationCalibration {
        reference = reference.clone();
    }

    @Override
    public double[] reference() {
        return reference.clone();
    }

    static ModelExplanationCalibration read(JsonNode root) {
        var node = root.path("explanation");
        if (node.isMissingNode() || node.isNull()) {
            return null;
        }
        require(node.isObject() && METHOD.equals(node.path("method").textValue()) && node.path("reference").isObject());
        double[] reference = new double[ModelArtifactMetadata.FEATURE_ORDER.size()];
        for (int i = 0; i < reference.length; i++) {
            var value = node.path("reference").path(ModelArtifactMetadata.FEATURE_ORDER.get(i));
            require(value.isNumber() && Double.isFinite(value.doubleValue()) && Math.abs(value.doubleValue()) <= Float.MAX_VALUE);
            reference[i] = value.doubleValue();
        }
        require(node.path("reference").size() == reference.length);
        return new ModelExplanationCalibration(METHOD, reference);
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw new IllegalArgumentException("Incompatible model artifact metadata");
        }
    }
}
