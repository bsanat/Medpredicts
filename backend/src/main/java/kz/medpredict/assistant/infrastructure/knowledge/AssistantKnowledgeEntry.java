package kz.medpredict.assistant.infrastructure.knowledge;

/** One approved fact with the project document it was taken from. */
public record AssistantKnowledgeEntry(String id, String title, String text, String source) {
}
