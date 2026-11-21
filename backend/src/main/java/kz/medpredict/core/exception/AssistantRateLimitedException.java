package kz.medpredict.core.exception;

import kz.medpredict.core.i18n.MessageCode;
import kz.medpredict.core.util.ErrorCode;

public class AssistantRateLimitedException extends ServiceValidationException {
    public AssistantRateLimitedException() {
        super(ErrorCode.ASSISTANT_RATE_LIMITED, MessageCode.ASSISTANT_RATE_LIMITED);
    }
}
