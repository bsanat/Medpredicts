package kz.medpredict.assistant.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import kz.medpredict.assistant.model.dto.AssistantMessageCreateRequest;
import kz.medpredict.assistant.model.dto.AssistantMessageResponse;
import kz.medpredict.assistant.model.dto.AssistantStatusResponse;
import kz.medpredict.assistant.service.AssistantService;
import kz.medpredict.core.exception.dto.ApiErrorResponse;
import kz.medpredict.core.exception.dto.ApiValidationErrorResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/assistant")
@Tag(name = "Assistant", description = "Documentation assistant grounded in approved project facts; separate from prediction")
public class AssistantController {
    private final AssistantService service;

    @GetMapping("/status")
    @Operation(operationId = "getAssistantStatus", summary = "Whether the assistant is configured")
    public AssistantStatusResponse status() {
        return service.status();
    }

    @PostMapping("/messages")
    @Operation(operationId = "createAssistantMessage", summary = "Ask the documentation assistant",
            description = "The question and optional derived result numbers are sent to an external language model (Google Gemini). "
                    + "Raw medical inputs are not part of the request contract. The assistant never diagnoses or recommends treatment.")
    @ApiResponse(responseCode = "200", description = "Grounded answer or an explicit refusal")
    @ApiResponse(responseCode = "400", description = "Invalid message, language or context",
            content = @Content(schema = @Schema(implementation = ApiValidationErrorResponse.class)))
    @ApiResponse(responseCode = "429", description = "Assistant rate limit reached",
            content = @Content(schema = @Schema(implementation = ApiErrorResponse.class),
                    examples = @ExampleObject(value = "{\"error\":\"ASSISTANT_RATE_LIMITED\",\"message\":\"Too many assistant requests. Please wait a minute.\"}")))
    @ApiResponse(responseCode = "503", description = "Assistant not configured or provider unavailable",
            content = @Content(schema = @Schema(implementation = ApiErrorResponse.class),
                    examples = @ExampleObject(value = "{\"error\":\"ASSISTANT_UNAVAILABLE\",\"message\":\"The assistant is not available right now.\"}")))
    public AssistantMessageResponse create(@Valid @RequestBody AssistantMessageCreateRequest request) {
        return service.reply(request);
    }
}
