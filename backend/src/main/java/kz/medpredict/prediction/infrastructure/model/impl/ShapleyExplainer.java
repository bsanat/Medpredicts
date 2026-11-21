package kz.medpredict.prediction.infrastructure.model.impl;

import java.util.ArrayList;
import java.util.List;
import kz.medpredict.prediction.infrastructure.model.ModelExplanationCalibration;
import kz.medpredict.prediction.model.dto.PredictionContributionResponse;
import kz.medpredict.prediction.model.dto.PredictionExplanationResponse;

/**
 * Exact baseline Shapley values over all feature coalitions. Row k of the coalition batch takes the input value for
 * feature i when bit (n-1-i) of k is set and the reference value otherwise, matching Python's itertools.product order.
 */
final class ShapleyExplainer {
    private ShapleyExplainer() {
    }

    static float[][] coalitions(float[] input, double[] reference) {
        int n = input.length;
        float[][] rows = new float[1 << n][n];
        for (int k = 0; k < rows.length; k++) {
            for (int i = 0; i < n; i++) {
                rows[k][i] = present(k, i, n) ? input[i] : (float) reference[i];
            }
        }
        return rows;
    }

    static PredictionExplanationResponse explain(float[] outputs, List<String> keys) {
        int n = keys.size();
        if (outputs.length != 1 << n) {
            throw new IllegalArgumentException("Coalition outputs do not match feature count");
        }
        List<PredictionContributionResponse> contributions = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double phi = 0;
            for (int k = 0; k < outputs.length; k++) {
                if (present(k, i, n)) {
                    continue;
                }
                int size = Integer.bitCount(k);
                phi += weight(size, n) * ((double) outputs[k | bit(i, n)] - outputs[k]);
            }
            contributions.add(new PredictionContributionResponse(keys.get(i), phi));
        }
        return new PredictionExplanationResponse(outputs[0], contributions, ModelExplanationCalibration.METHOD);
    }

    private static boolean present(int coalition, int feature, int n) {
        return (coalition & bit(feature, n)) != 0;
    }

    private static int bit(int feature, int n) {
        return 1 << (n - 1 - feature);
    }

    private static double weight(int size, int n) {
        return factorial(size) * factorial(n - size - 1) / factorial(n);
    }

    private static double factorial(int value) {
        double result = 1;
        for (int i = 2; i <= value; i++) {
            result *= i;
        }
        return result;
    }
}
