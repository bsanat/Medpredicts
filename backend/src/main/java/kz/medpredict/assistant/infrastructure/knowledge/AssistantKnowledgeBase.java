package kz.medpredict.assistant.infrastructure.knowledge;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/** Immutable approved knowledge loaded once from the classpath; startup fails on an invalid file. */
@Component
public class AssistantKnowledgeBase {
    static final String RESOURCE = "/assistant/knowledge-base.json";
    private final String version;
    private final Map<String, AssistantKnowledgeEntry> entries;
    private final List<AssistantKnowledgeEntry> ordered;

    @Autowired
    public AssistantKnowledgeBase(ObjectMapper objectMapper) {
        this(read(objectMapper));
    }

    AssistantKnowledgeBase(JsonNode root) {
        require(root != null && root.path("version").isTextual() && root.path("entries").isArray()
                && !root.path("entries").isEmpty());
        Map<String, AssistantKnowledgeEntry> loaded = new LinkedHashMap<>();
        for (JsonNode node : root.path("entries")) {
            var entry = new AssistantKnowledgeEntry(text(node, "id"), text(node, "title"), text(node, "text"), text(node, "source"));
            require(entry.id().matches("[a-z][a-z0-9-]{1,40}") && loaded.putIfAbsent(entry.id(), entry) == null);
        }
        this.version = root.path("version").textValue();
        this.entries = Map.copyOf(loaded);
        this.ordered = List.copyOf(loaded.values());
    }


    public String version() {
        return version;
    }

    public List<AssistantKnowledgeEntry> entries() {
        return ordered;
    }

    public Optional<AssistantKnowledgeEntry> find(String id) {
        return Optional.ofNullable(id == null ? null : entries.get(id));
    }

    private static JsonNode read(ObjectMapper objectMapper) {
        try (InputStream stream = AssistantKnowledgeBase.class.getResourceAsStream(RESOURCE)) {
            require(stream != null);
            return objectMapper.readTree(stream);
        } catch (IOException exception) {
            throw new IllegalStateException("Assistant knowledge base cannot be read", exception);
        }
    }

    private static String text(JsonNode node, String key) {
        require(node.path(key).isTextual() && !node.path(key).textValue().isBlank());
        return node.path(key).textValue();
    }

    private static void require(boolean condition) {
        if (!condition) {
            throw new IllegalStateException("Invalid assistant knowledge base");
        }
    }
}
