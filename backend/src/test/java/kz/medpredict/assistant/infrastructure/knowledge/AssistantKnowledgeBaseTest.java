package kz.medpredict.assistant.infrastructure.knowledge;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AssistantKnowledgeBaseTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void approvedKnowledgeLoadsWithUniqueIdsAndExistingSourceDocuments() {
        var knowledge = new AssistantKnowledgeBase(mapper);
        assertThat(knowledge.entries()).hasSizeGreaterThanOrEqualTo(20);
        assertThat(knowledge.find("not-diagnosis")).isPresent();
        assertThat(knowledge.find("missing")).isEmpty();
        var repository = Path.of("..").toAbsolutePath().normalize();
        if (Files.isDirectory(repository.resolve("ml-ai"))) {   // monorepo checkout: every cited document must exist
            knowledge.entries().forEach(entry -> assertThat(repository.resolve(entry.source())).as(entry.id()).exists());
        }
    }

    @Test
    void rejectsDuplicateIdsAndMissingFields() throws Exception {
        for (String invalid : new String[]{
                "{\"version\":\"v\",\"entries\":[{\"id\":\"a-b\",\"title\":\"t\",\"text\":\"x\",\"source\":\"s\"},{\"id\":\"a-b\",\"title\":\"t\",\"text\":\"x\",\"source\":\"s\"}]}",
                "{\"version\":\"v\",\"entries\":[{\"id\":\"a-b\",\"title\":\"\",\"text\":\"x\",\"source\":\"s\"}]}",
                "{\"version\":\"v\",\"entries\":[]}", "{\"entries\":[]}"}) {
            assertThatThrownBy(() -> new AssistantKnowledgeBase(mapper.readTree(invalid))).isInstanceOf(IllegalStateException.class);
        }
    }
}
