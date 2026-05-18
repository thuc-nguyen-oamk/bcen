package webapi.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import springfox.documentation.annotations.ApiIgnore;
import vn.pipeline.Annotation;
import webapi.service.NLPService;

import java.util.*;

/**
 * REST Controller for VnCoreNLP processing endpoints.
 */
@RestController
@RequestMapping("/api/v1")
@CrossOrigin(origins = "*")
public class NLPController {
    
    @Autowired
    private NLPService nlpService;
    
    /**
     * Process Vietnamese text with specified annotations.
     * 
     * @param request Processing request with text and annotators
     * @return Processing result in JSON format
     */
    @PostMapping(value = "/process", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Map<String, Object>> process(@RequestBody ProcessRequest request) {
        try {
            if (request.getText() == null || request.getText().trim().isEmpty()) {
                return ResponseEntity.badRequest().body(Map.of(
                    "error", "Text field is required"
                ));
            }
            
            List<String> annotators = request.getAnnotators();
            if (annotators == null || annotators.isEmpty()) {
                annotators = Arrays.asList("wseg", "pos", "ner", "parse");
            }
            
            Annotation annotation = nlpService.process(request.getText(), annotators);
            Map<String, Object> result = nlpService.annotationToMap(annotation);
            result.put("annotators", annotators);
            
            return ResponseEntity.ok(result);
            
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of(
                "error", "Processing failed: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Process text and return tabular format.
     */
    @PostMapping(value = "/process/tabular", produces = MediaType.TEXT_PLAIN_VALUE)
    public ResponseEntity<String> processTabular(@RequestBody ProcessRequest request) {
        try {
            if (request.getText() == null || request.getText().trim().isEmpty()) {
                return ResponseEntity.badRequest().body("Text field is required");
            }
            
            List<String> annotators = request.getAnnotators();
            if (annotators == null || annotators.isEmpty()) {
                annotators = Arrays.asList("wseg", "pos", "ner", "parse");
            }
            
            Annotation annotation = nlpService.process(request.getText(), annotators);
            return ResponseEntity.ok(nlpService.annotationToTabular(annotation));
            
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error: " + e.getMessage());
        }
    }
    
    /**
     * Extract named entities from text.
     */
    @PostMapping(value = "/entities", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<List<Map<String, Object>>> extractEntities(@RequestBody ProcessRequest request) {
        try {
            if (request.getText() == null || request.getText().trim().isEmpty()) {
                return ResponseEntity.badRequest().body(List.of(Map.of("error", "Text field is required")));
            }
            
            List<String> annotators = Arrays.asList("wseg", "pos", "ner");
            Annotation annotation = nlpService.process(request.getText(), annotators);
            List<Map<String, Object>> entities = nlpService.extractEntities(annotation);
            
            return ResponseEntity.ok(entities);
            
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(List.of(Map.of("error", e.getMessage())));
        }
    }
    
    /**
     * Health check endpoint.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        return ResponseEntity.ok(Map.of(
            "status", "UP",
            "service", "VnCoreNLP API",
            "version", "1.0.0",
            "timestamp", System.currentTimeMillis()
        ));
    }
    
    /**
     * Get available annotators.
     */
    @GetMapping("/annotators")
    public ResponseEntity<List<Map<String, String>>> getAnnotators() {
        List<Map<String, String>> annotators = Arrays.asList(
            Map.of("name", "wseg", "description", "Word Segmentation"),
            Map.of("name", "pos", "description", "Part-of-Speech Tagging"),
            Map.of("name", "ner", "description", "Named Entity Recognition"),
            Map.of("name", "parse", "description", "Dependency Parsing")
        );
        return ResponseEntity.ok(annotators);
    }
    
    /**
     * Request body for processing.
     */
    public static class ProcessRequest {
        private String text;
        private List<String> annotators;
        
        public String getText() {
            return text;
        }
        
        public void setText(String text) {
            this.text = text;
        }
        
        public List<String> getAnnotators() {
            return annotators;
        }
        
        public void setAnnotators(List<String> annotators) {
            this.annotators = annotators;
        }
    }
}
