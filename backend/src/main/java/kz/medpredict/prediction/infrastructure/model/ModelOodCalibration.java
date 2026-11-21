package kz.medpredict.prediction.infrastructure.model;

import com.fasterxml.jackson.databind.JsonNode;

/** Squared Mahalanobis distance against training feature statistics; above the threshold the input is atypical. */
public record ModelOodCalibration(double[] mean, double[][] precision, double threshold) {
    public static final String WARNING = "OUT_OF_DISTRIBUTION";

    public ModelOodCalibration {
        mean = mean.clone();
        precision = copy(precision);
    }

    @Override
    public double[] mean() {
        return mean.clone();
    }

    @Override
    public double[][] precision() {
        return copy(precision);
    }

    public double squaredDistance(double[] features) {
        int n = mean.length;
        double[] delta = new double[n];
        for (int i = 0; i < n; i++) {
            delta[i] = features[i] - mean[i];
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += delta[i] * precision[i][j] * delta[j];
            }
        }
        return sum;
    }

    public boolean isOutOfDistribution(double[] features) {
        return squaredDistance(features) > threshold;
    }

    static ModelOodCalibration read(JsonNode root) {
        var node = root.path("outOfDistribution");
        if (node.isMissingNode() || node.isNull()) {
            return null;
        }
        int n = ModelArtifactMetadata.FEATURE_ORDER.size();
        require(node.isObject() && "mahalanobis_squared".equals(node.path("method").textValue()));
        double[] mean = vector(node.path("mean"), n);
        require(node.path("precision").isArray() && node.path("precision").size() == n);
        double[][] precision = new double[n][];
        for (int i = 0; i < n; i++) {
            precision[i] = vector(node.path("precision").get(i), n);
        }
        for (int i = 0; i < n; i++) {
            require(precision[i][i] > 0);
            for (int j = 0; j < n; j++) {
                require(Math.abs(precision[i][j] - precision[j][i]) <= 1e-9 * Math.max(1, Math.abs(precision[i][j])));
            }
        }
        var threshold = node.path("threshold");
        require(threshold.isNumber() && Double.isFinite(threshold.doubleValue()) && threshold.doubleValue() > 0);
        return new ModelOodCalibration(mean, precision, threshold.doubleValue());
    }

    private static double[] vector(JsonNode node, int size) {
        require(node.isArray() && node.size() == size);
        double[] values = new double[size];
        for (int i = 0; i < size; i++) {
            require(node.get(i).isNumber() && Double.isFinite(node.get(i).doubleValue()));
            values[i] = node.get(i).doubleValue();
        }
        return values;
    }

    private static double[][] copy(double[][] matrix) {
        double[][] result = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = matrix[i].clone();
        }
        return result;
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw new IllegalArgumentException("Incompatible model artifact metadata");
        }
    }
}
