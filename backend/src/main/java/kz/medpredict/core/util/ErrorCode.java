package kz.medpredict.core.util;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;

@Getter
@RequiredArgsConstructor
public enum ErrorCode {
    VALIDATION_ERROR(HttpStatus.BAD_REQUEST),
    INVALID_REQUEST(HttpStatus.BAD_REQUEST),
    MODEL_UNAVAILABLE(HttpStatus.SERVICE_UNAVAILABLE),
    ASSISTANT_UNAVAILABLE(HttpStatus.SERVICE_UNAVAILABLE),
    ASSISTANT_RATE_LIMITED(HttpStatus.TOO_MANY_REQUESTS),
    SYSTEM_ERROR(HttpStatus.INTERNAL_SERVER_ERROR);

    private final HttpStatus status;
}
