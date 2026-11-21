package kz.medpredict.core.exception;

import kz.medpredict.core.i18n.MessageCode;
import kz.medpredict.core.util.ErrorCode;

public class AssistantUnavailableException extends ServiceValidationException {
    public AssistantUnavailableException() {
        super(ErrorCode.ASSISTANT_UNAVAILABLE, MessageCode.ASSISTANT_UNAVAILABLE);
    }
}
