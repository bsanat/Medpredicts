package kz.medpredict.prediction.service.impl;

import jakarta.validation.Validator;
import kz.medpredict.core.exception.InvalidModelResponseException;
import kz.medpredict.core.exception.InvalidRequestException;
import kz.medpredict.prediction.infrastructure.model.PredictionModel;
import kz.medpredict.prediction.model.dto.PredictionCreateRequest;
import kz.medpredict.prediction.model.dto.PredictionModelInfoResponse;
import kz.medpredict.prediction.model.dto.PredictionResultResponse;
import kz.medpredict.prediction.model.dto.PredictionModelStatusResponse;
import kz.medpredict.prediction.service.PredictionService;
import kz.medpredict.prediction.validation.InputValidator;
import kz.medpredict.prediction.validation.ModelMetadataValidator;
import java.util.Set;
import kz.medpredict.prediction.infrastructure.model.ModelOodCalibration;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class PredictionServiceImpl implements PredictionService {
    private static final Set<String> KNOWN_WARNINGS = Set.of(ModelOodCalibration.WARNING);
    private static final double ADDITIVITY_TOLERANCE = 0.01;
    private final PredictionModel model;
    private final InputValidator inputValidator;
    private final Validator beanValidator;
    private final ModelMetadataValidator metadataValidator;

    @Override
    public PredictionModelStatusResponse modelStatus() {
        return model.status();
    }

    @Override
    public PredictionModelInfoResponse modelInfo() {
        var metadata = model.info();
        metadataValidator.validate(metadata);
        return metadata;
    }

    @Override
    public PredictionResultResponse predict(PredictionCreateRequest request) {
        if (request == null) {
            throw new InvalidRequestException();
        }
        var metadata = modelInfo();
        var input = inputValidator.validate(request.getValues(), metadata);
        var prediction = model.predict(input);
        validatePrediction(prediction, metadata.getUnit());
        return prediction;
    }

    private void validatePrediction(PredictionResultResponse prediction, String expectedUnit) {
        if (prediction == null || !Double.isFinite(prediction.getPredictedValue())
                || !expectedUnit.equals(prediction.getUnit())
                || !beanValidator.validate(prediction).isEmpty() || !validInterval(prediction)
                || !validExplanation(prediction) || !KNOWN_WARNINGS.containsAll(prediction.getWarnings())) {
            throw new InvalidModelResponseException();
        }
    }

    private static boolean validExplanation(PredictionResultResponse prediction) {
        var explanation = prediction.getExplanation();
        if (explanation == null) {
            return true;
        }
        if (!Double.isFinite(explanation.getReferenceValue()) || explanation.getContributions().isEmpty()) {
            return false;
        }
        double sum = explanation.getReferenceValue();
        for (var contribution : explanation.getContributions()) {
            if (contribution.getKey() == null || !Double.isFinite(contribution.getValue())) {
                return false;
            }
            sum += contribution.getValue();
        }
        // float32 graph outputs: allow small accumulation error in the exact decomposition.
        return Math.abs(sum - prediction.getPredictedValue()) <= ADDITIVITY_TOLERANCE;
    }

    private static boolean validInterval(PredictionResultResponse prediction) {
        var interval = prediction.getInterval();
        return interval == null || (Double.isFinite(interval.getLower()) && Double.isFinite(interval.getUpper())
                && interval.getLower() <= prediction.getPredictedValue() && prediction.getPredictedValue() <= interval.getUpper()
                && interval.getLevel() > 0 && interval.getLevel() < 1
                && interval.getMeasuredCoverage() >= 0 && interval.getMeasuredCoverage() <= 1);
    }

}
