package batch;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.commons.cli.*;
import vn.pipeline.*;

import java.io.*;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * VnCoreNLP Batch Processor
 * A high-performance command-line tool for processing multiple Vietnamese text files
 * with support for parallel processing and multiple output formats.
 */
public class BatchProcessor {
    
    private static Options options;
    private static VnCoreNLP pipeline;
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().build();
    
    public static void main(String[] args) {
        createOptions();
        
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        
        try {
            CommandLine cmd = parser.parse(options, args);
            
            if (cmd.hasOption("help")) {
                printHelp(formatter);
                return;
            }
            
            if (!cmd.hasOption("input") || !cmd.hasOption("output")) {
                System.err.println("Error: Input and output directories are required.");
                printHelp(formatter);
                System.exit(1);
            }
            
            String inputDir = cmd.getOptionValue("input");
            String outputDir = cmd.getOptionValue("output");
            String format = cmd.getOptionValue("format", "tabular");
            int threads = Integer.parseInt(cmd.getOptionValue("threads", "4"));
            boolean wseg = !cmd.hasOption("disable-wseg");
            boolean pos = !cmd.hasOption("disable-pos");
            boolean ner = !cmd.hasOption("disable-ner");
            boolean parse = !cmd.hasOption("disable-parse");
            
            System.out.println("╔════════════════════════════════════════════════════════╗");
            System.out.println("║     VnCoreNLP Batch Processor v1.0                     ║");
            System.out.println("╚════════════════════════════════════════════════════════╝");
            System.out.println();
            
            // Initialize pipeline
            System.out.println("🔄 Initializing NLP pipeline...");
            List<String> annotatorsList = new ArrayList<>();
            if (wseg) annotatorsList.add("wseg");
            if (pos) annotatorsList.add("pos");
            if (ner) annotatorsList.add("ner");
            if (parse) annotatorsList.add("parse");
            
            String[] annotators = annotatorsList.toArray(new String[0]);
            long startTime = System.currentTimeMillis();
            pipeline = new VnCoreNLP(annotators);
            System.out.println("✓ Pipeline initialized in " + (System.currentTimeMillis() - startTime) + "ms");
            System.out.println();
            
            // Find all text files
            List<Path> files = findTextFiles(inputDir);
            if (files.isEmpty()) {
                System.err.println("No text files found in: " + inputDir);
                System.exit(1);
            }
            
            System.out.println("📁 Found " + files.size() + " file(s) to process");
            System.out.println("🧵 Using " + threads + " thread(s)");
            System.out.println("📄 Output format: " + format);
            System.out.println();
            
            // Create output directory
            Files.createDirectories(Paths.get(outputDir));
            
            // Process files in parallel
            processFilesParallel(files, outputDir, format, threads);
            
            System.out.println();
            System.out.println("╔════════════════════════════════════════════════════════╗");
            System.out.println("║     Processing completed successfully!                 ║");
            System.out.println("╚════════════════════════════════════════════════════════╝");
            
        } catch (ParseException e) {
            System.err.println("Error parsing command line: " + e.getMessage());
            printHelp(formatter);
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static void createOptions() {
        options = new Options();
        
        Option input = Option.builder("i")
            .longOpt("input")
            .argName("DIR")
            .required(false)
            .desc("Input directory containing text files")
            .build();
        options.addOption(input);
        
        Option output = Option.builder("o")
            .longOpt("output")
            .argName("DIR")
            .required(false)
            .desc("Output directory for processed results")
            .build();
        options.addOption(output);
        
        Option format = Option.builder("f")
            .longOpt("format")
            .argName("FORMAT")
            .desc("Output format: tabular, json, conll (default: tabular)")
            .build();
        options.addOption(format);
        
        Option threads = Option.builder("t")
            .longOpt("threads")
            .argName("NUM")
            .desc("Number of parallel threads (default: 4)")
            .build();
        options.addOption(threads);
        
        options.addOption(Option.builder().longOpt("disable-wseg").desc("Disable word segmentation").build());
        options.addOption(Option.builder().longOpt("disable-pos").desc("Disable POS tagging").build());
        options.addOption(Option.builder().longOpt("disable-ner").desc("Disable NER").build());
        options.addOption(Option.builder().longOpt("disable-parse").desc("Disable dependency parsing").build());
        
        options.addOption(Option.builder("h").longOpt("help").desc("Print this help message").build());
    }
    
    private static void printHelp(HelpFormatter formatter) {
        formatter.setWidth(80);
        formatter.printHelp("VnCoreNLP-BatchProcessor [OPTIONS]", options);
        System.out.println();
        System.out.println("Examples:");
        System.out.println("  java -jar VnCoreNLP-BatchProcessor.jar -i ./input -o ./output");
        System.out.println("  java -jar VnCoreNLP-BatchProcessor.jar -i ./input -o ./output -f json -t 8");
        System.out.println("  java -jar VnCoreNLP-BatchProcessor.jar --input ./input --output ./output --format conll");
    }
    
    private static List<Path> findTextFiles(String dirPath) throws IOException {
        List<Path> files = new ArrayList<>();
        Path startPath = Paths.get(dirPath);
        
        if (!Files.exists(startPath)) {
            throw new IOException("Input directory does not exist: " + dirPath);
        }
        
        Files.walkFileTree(startPath, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                String fileName = file.getFileName().toString().toLowerCase();
                if (fileName.endsWith(".txt") || fileName.endsWith(".utf8") || 
                    fileName.endsWith(".vn") || fileName.endsWith(".vnt")) {
                    files.add(file);
                }
                return FileVisitResult.CONTINUE;
            }
        });
        
        return files;
    }
    
    private static void processFilesParallel(List<Path> files, String outputDir, 
                                            String format, int numThreads) {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        CompletionService<Void> completionService = new ExecutorCompletionService<>(executor);
        
        AtomicInteger processed = new AtomicInteger(0);
        AtomicInteger failed = new AtomicInteger(0);
        long totalWords = 0;
        long totalSentences = 0;
        
        // Submit tasks
        for (Path file : files) {
            completionService.submit(() -> {
                try {
                    processFile(file, outputDir, format);
                    processed.incrementAndGet();
                    
                    // Print progress
                    int current = processed.get();
                    int total = files.size();
                    double percent = (current * 100.0) / total;
                    System.out.printf("\r📊 Progress: [%d/%d] %.1f%% (%d failed)", 
                        current, total, percent, failed.get());
                    
                } catch (Exception e) {
                    failed.incrementAndGet();
                    System.err.println("\n✗ Error processing " + file + ": " + e.getMessage());
                }
                return null;
            });
        }
        
        // Wait for all tasks to complete
        for (int i = 0; i < files.size(); i++) {
            try {
                completionService.take().get();
            } catch (InterruptedException | ExecutionException e) {
                failed.incrementAndGet();
            }
        }
        
        executor.shutdown();
        System.out.println();
        System.out.println("✓ Processed: " + processed.get() + " files");
        if (failed.get() > 0) {
            System.out.println("✗ Failed: " + failed.get() + " files");
        }
    }
    
    private static void processFile(Path inputFile, String outputDir, String format) throws IOException {
        // Read input file
        String content = new String(Files.readAllBytes(inputFile));
        
        // Create annotation
        Annotation annotation = new Annotation(content);
        pipeline.annotate(annotation);
        
        // Format output
        String output;
        switch (format.toLowerCase()) {
            case "json":
                output = formatAsJSON(annotation);
                break;
            case "conll":
                output = formatAsCoNLL(annotation);
                break;
            default:
                output = annotation.toString();
        }
        
        // Write output file
        String relativePath = inputFile.getParent().relativize(inputFile).toString();
        String outputFileName = relativePath.replace(File.separator, "_");
        
        String extension = "." + (format.equals("json") ? "json" : "txt");
        Path outputPath = Paths.get(outputDir, outputFileName + extension);
        
        Files.write(outputPath, output.getBytes());
    }
    
    private static String formatAsJSON(Annotation annotation) {
        Map<String, Object> result = new LinkedHashMap<>();
        List<Map<String, Object>> sentencesList = new ArrayList<>();
        
        for (int i = 0; i < annotation.getSentences().size(); i++) {
            Sentence sentence = annotation.getSentences().get(i);
            Map<String, Object> sentenceMap = new LinkedHashMap<>();
            sentenceMap.put("id", i + 1);
            
            List<Map<String, Object>> wordsList = new ArrayList<>();
            for (Word word : sentence.getWords()) {
                Map<String, Object> wordMap = new LinkedHashMap<>();
                wordMap.put("text", word.getForm());
                wordMap.put("pos", word.getPos());
                wordMap.put("ner", word.getNer());
                wordMap.put("head", word.getHead());
                wordMap.put("dep", word.getDep());
                wordsList.add(wordMap);
            }
            
            sentenceMap.put("words", wordsList);
            sentencesList.add(sentenceMap);
        }
        
        result.put("sentences", sentencesList);
        return gson.toJson(result);
    }
    
    private static String formatAsCoNLL(Annotation annotation) {
        StringBuilder sb = new StringBuilder();
        
        for (Sentence sentence : annotation.getSentences()) {
            for (Word word : sentence.getWords()) {
                sb.append(word.getId())
                  .append("\t")
                  .append(word.getForm())
                  .append("\t")
                  .append("_")  // lemma
                  .append("\t")
                  .append(word.getPos())
                  .append("\t")
                  .append("_")  // fine-grained POS
                  .append("\t")
                  .append("_")  // features
                  .append("\t")
                  .append(word.getHead())
                  .append("\t")
                  .append(word.getDep())
                  .append("\t")
                  .append("_")  // deps
                  .append("\t")
                  .append(word.getNer().equals("O") ? "_" : word.getNer())
                  .append("\n");
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
}
