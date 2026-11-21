package kz.medpredict.core.i18n;

public interface MessageService {
    String getMessage(String code, Object... args);
    String getMessage(String code);

    /** Translates for an explicitly chosen supported language instead of the request locale. */
    String getMessageForLanguage(String code, java.util.Locale locale);
}
