package webapi.service;

import org.springframework.stereotype.Service;
import vn.pipeline.*;

import javax.annotation.PostConstruct;
import java.util.*;

/**
 * Service class that wraps VnCoreNLP pipeline functionality.
 */
@Service
public class NLPService {
    
    private VnCoreNLP fullPipeline;
    private VnCoreNLP wsegPipeline;
    private VnCoreNLP posPipeline;
    private VnCoreNLP nerPipeline;
    private VnCoreNLP parsePipeline;
    
    @PostConstruct
    public void init() throws Exception {
        // Initialize pipelines for different annotation combinations
        fullPipeline = new VnCoreNLP(new String[]{"wseg", "pos", "ner", "parse"});
        wsegPipeline = new VnCoreNLP(new String[]{"wseg"});
        posPipeline = new VnCoreNLP(new String[]{"wseg", "pos"});
        nerPipeline = new VnCoreNLP(new String[]{"wseg", "pos", "ner"});
        parsePipeline = fullPipeline; // parsing requires all previous steps
    }
    
    /**
     * Process text with specified annotations.
     */
    public Annotation process(String text, List<String> annotators) throws Exception {
        String[] annotatorArray = annotators.toArray(new String[0]);
        VnCoreNLP pipeline = getPipeline(annotators);
        
        Annotation annotation = new Annotation(text);
        pipeline.annotate(annotation);
        
        return annotation;
    }
    
    /**
     * Get the appropriate pipeline based on requested annotators.
     */
    private VnCoreNLP getPipeline(List<String> annotators) {
        if (annotators.contains("parse")) {
            return parsePipeline;
        } else if (annotators.contains("ner")) {
            return nerPipeline;
        } else if (annotators.contains("pos")) {
            return posPipeline;
        } else {
            return wsegPipeline;
        }
    }
    
    /**
     * Convert Annotation to structured map for JSON response.
     */
    public Map<String, Object> annotationToMap(Annotation annotation) {
        Map<String, Object> result = new LinkedHashMap<>();
        List<Map<String, Object>> sentencesList = new ArrayList<>();
        
        int totalWords = 0;
        int totalSentences = annotation.getSentences().size();
        
        for (int i = 0; i < annotation.getSentences().size(); i++) {
            Sentence sentence = annotation.getSentences().get(i);
            Map<String, Object> sentenceMap = new LinkedHashMap<>();
            sentenceMap.put("sentenceId", i + 1);
            
            List<Map<String, Object>> wordsList = new ArrayList<>();
            for (Word word : sentence.getWords()) {
                Map<String, Object> wordMap = new LinkedHashMap<>();
                wordMap.put("id", word.getId());
                wordMap.put("text", word.getForm());
                wordMap.put("pos", word.getPos());
                wordMap.put("ner", word.getNer());
                wordMap.put("head", word.getHead());
                wordMap.put("dep", word.getDep());
                wordsList.add(wordMap);
                totalWords++;
            }
            
            sentenceMap.put("words", wordsList);
            sentencesList.add(sentenceMap);
        }
        
        result.put("sentences", sentencesList);
        result.put("statistics", Map.of(
            "totalSentences", totalSentences,
            "totalWords", totalWords
        ));
        
        return result;
    }
    
    /**
     * Convert Annotation to tabular string format.
     */
    public String annotationToTabular(Annotation annotation) {
        return annotation.toString();
    }
    
    /**
     * Extract named entities from annotation.
     */
    public List<Map<String, Object>> extractEntities(Annotation annotation) {
        List<Map<String, Object>> entities = new ArrayList<>();
        
        for (Sentence sentence : annotation.getSentences()) {
            StringBuilder currentEntity = new StringBuilder();
            String currentType = null;
            int startIndex = -1;
            
            for (Word word : sentence.getWords()) {
                String ner = word.getNer();
                
                if (ner.startsWith("B-")) {
                    // Save previous entity if exists
                    if (currentEntity.length() > 0 && currentType != null) {
                        entities.add(createEntity(currentEntity.toString(), currentType, startIndex));
                    }
                    // Start new entity
                    currentEntity = new StringBuilder(word.getForm());
                    currentType = ner.substring(2);
                    startIndex = word.getId();
                } else if (ner.startsWith("I-") && currentEntity.length() > 0) {
                    // Continue current entity
                    currentEntity.append(" ").append(word.getForm());
                } else {
                    // End current entity if exists
                    if (currentEntity.length() > 0 && currentType != null) {
                        entities.add(createEntity(currentEntity.toString(), currentType, startIndex));
                        currentEntity = new StringBuilder();
                        currentType = null;
                    }
                }
            }
            
            // Don't forget last entity in sentence
            if (currentEntity.length() > 0 && currentType != null) {
                entities.add(createEntity(currentEntity.toString(), currentType, startIndex));
            }
        }
        
        return entities;
    }
    
    private Map<String, Object> createEntity(String text, String type, int index) {
        Map<String, Object> entity = new LinkedHashMap<>();
        entity.put("text", text);
        entity.put("type", type);
        entity.put("position", index);
        return entity;
    }
}
