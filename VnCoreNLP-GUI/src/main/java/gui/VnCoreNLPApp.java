package gui;

import vn.pipeline.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * VnCoreNLP Desktop Application
 * A comprehensive GUI for Vietnamese NLP processing featuring:
 * - Word Segmentation
 * - POS Tagging
 * - Named Entity Recognition
 * - Dependency Parsing
 */
public class VnCoreNLPApp extends JFrame {
    
    private JTextArea inputTextArea;
    private JTextArea outputTextArea;
    private JCheckBox wsegCheckBox;
    private JCheckBox posCheckBox;
    private JCheckBox nerCheckBox;
    private JCheckBox parseCheckBox;
    private JLabel statusLabel;
    private JLabel statsLabel;
    private JButton processButton;
    private JButton clearButton;
    private JButton copyButton;
    private JButton saveButton;
    private JButton loadButton;
    private JComboBox<String> outputFormatCombo;
    private JProgressBar progressBar;
    
    private VnCoreNLP pipeline;
    private boolean isProcessing = false;
    
    public VnCoreNLPApp() {
        initPipeline();
        initComponents();
    }
    
    private void initPipeline() {
        SwingWorker<Void, Void> worker = new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() throws Exception {
                String[] annotators = {"wseg", "pos", "ner", "parse"};
                pipeline = new VnCoreNLP(annotators);
                return null;
            }
            
            @Override
            protected void done() {
                statusLabel.setText("✓ Pipeline loaded successfully");
                statusLabel.setForeground(new Color(0, 128, 0));
                processButton.setEnabled(true);
            }
        };
        worker.execute();
    }
    
    private void initComponents() {
        setTitle("VnCoreNLP - Vietnamese NLP Processing Tool");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1200, 800);
        setLocationRelativeTo(null);
        
        // Main panel with border
        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(new EmptyBorder(15, 15, 15, 15));
        
        // Title Panel
        JPanel titlePanel = createTitlePanel();
        mainPanel.add(titlePanel, BorderLayout.NORTH);
        
        // Control Panel
        JPanel controlPanel = createControlPanel();
        mainPanel.add(controlPanel, BorderLayout.CENTER);
        
        // Status Bar
        JPanel statusPanel = createStatusPanel();
        mainPanel.add(statusPanel, BorderLayout.SOUTH);
        
        add(mainPanel);
        
        // Apply modern look and feel
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        SwingUtilities.updateComponentTreeUI(this);
    }
    
    private JPanel createTitlePanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(new EmptyBorder(0, 0, 10, 0));
        
        JLabel titleLabel = new JLabel("🇻🇳 VnCoreNLP Desktop Application");
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 24));
        titleLabel.setForeground(new Color(41, 128, 185));
        
        JLabel subtitleLabel = new JLabel("Word Segmentation | POS Tagging | Named Entity Recognition | Dependency Parsing");
        subtitleLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        subtitleLabel.setForeground(Color.GRAY);
        
        panel.add(titleLabel, BorderLayout.WEST);
        panel.add(subtitleLabel, BorderLayout.SOUTH);
        
        return panel;
    }
    
    private JPanel createControlPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.insets = new Insets(5, 5, 5, 5);
        
        // Options Panel
        JPanel optionsPanel = createOptionsPanel();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        gbc.weightx = 1.0;
        gbc.weighty = 0.0;
        panel.add(optionsPanel, gbc);
        
        // Input Panel
        JPanel inputPanel = createInputPanel();
        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.gridwidth = 1;
        gbc.weightx = 0.5;
        gbc.weighty = 1.0;
        panel.add(inputPanel, gbc);
        
        // Output Panel
        JPanel outputPanel = createOutputPanel();
        gbc.gridx = 1;
        gbc.gridy = 1;
        gbc.gridwidth = 1;
        gbc.weightx = 0.5;
        gbc.weighty = 1.0;
        panel.add(outputPanel, gbc);
        
        return panel;
    }
    
    private JPanel createOptionsPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 10));
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createLineBorder(new Color(41, 128, 185)),
            " Processing Options ",
            javax.swing.border.TitledBorder.DEFAULT_JUSTIFICATION,
            javax.swing.border.TitledBorder.DEFAULT_POSITION,
            new Font("Segoe UI", Font.BOLD, 12),
            new Color(41, 128, 185)
        ));
        
        JLabel optionsLabel = new JLabel("Select annotations:");
        optionsLabel.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        panel.add(optionsLabel);
        
        wsegCheckBox = createCheckBox("Word Segmentation", true);
        panel.add(wsegCheckBox);
        
        posCheckBox = createCheckBox("POS Tagging", true);
        panel.add(posCheckBox);
        
        nerCheckBox = createCheckBox("Named Entity Recognition", true);
        panel.add(nerCheckBox);
        
        parseCheckBox = createCheckBox("Dependency Parsing", true);
        panel.add(parseCheckBox);
        
        panel.add(Box.createHorizontalStrut(20));
        
        JLabel formatLabel = new JLabel("Output Format:");
        formatLabel.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        panel.add(formatLabel);
        
        String[] formats = {"Tabular (Default)", "JSON", "Plain Text", "HTML"};
        outputFormatCombo = new JComboBox<>(formats);
        outputFormatCombo.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        outputFormatCombo.setPreferredSize(new Dimension(150, 25));
        panel.add(outputFormatCombo);
        
        return panel;
    }
    
    private JCheckBox createCheckBox(String text, boolean selected) {
        JCheckBox checkBox = new JCheckBox(text, selected);
        checkBox.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        checkBox.setFocusPainted(false);
        return checkBox;
    }
    
    private JPanel createInputPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createLineBorder(new Color(39, 174, 96)),
            " Input Text ",
            javax.swing.border.TitledBorder.DEFAULT_JUSTIFICATION,
            javax.swing.border.TitledBorder.DEFAULT_POSITION,
            new Font("Segoe UI", Font.BOLD, 12),
            new Color(39, 174, 96)
        ));
        
        // Toolbar
        JPanel toolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 5));
        
        loadButton = createButton("📂 Load", new Color(52, 152, 219));
        loadButton.addActionListener(e -> loadFile());
        toolbar.add(loadButton);
        
        JButton pasteButton = createButton("📋 Paste", new Color(52, 152, 219));
        pasteButton.addActionListener(e -> pasteFromClipboard());
        toolbar.add(pasteButton);
        
        clearButton = createButton("🗑 Clear", new Color(231, 76, 60));
        clearButton.addActionListener(e -> inputTextArea.setText(""));
        toolbar.add(clearButton);
        
        panel.add(toolbar, BorderLayout.NORTH);
        
        // Text Area
        inputTextArea = new JTextArea();
        inputTextArea.setFont(new Font("Consolas", Font.PLAIN, 14));
        inputTextArea.setLineWrap(true);
        inputTextArea.setWrapStyleWord(true);
        inputTextArea.setBorder(new EmptyBorder(10, 10, 10, 10));
        inputTextArea.setBackground(new Color(250, 250, 250));
        
        JScrollPane scrollPane = new JScrollPane(inputTextArea);
        scrollPane.setBorder(BorderFactory.createLineBorder(new Color(200, 200, 200)));
        panel.add(scrollPane, BorderLayout.CENTER);
        
        // Character count
        JLabel charCountLabel = new JLabel("Characters: 0");
        charCountLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        charCountLabel.setForeground(Color.GRAY);
        
        inputTextArea.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                updateCharCount(charCountLabel);
            }
            
            @Override
            public void removeUpdate(DocumentEvent e) {
                updateCharCount(charCountLabel);
            }
            
            @Override
            public void changedUpdate(DocumentEvent e) {
                updateCharCount(charCountLabel);
            }
            
            private void updateCharCount(JLabel label) {
                label.setText("Characters: " + inputTextArea.getText().length());
            }
        });
        
        panel.add(charCountLabel, BorderLayout.SOUTH);
        
        return panel;
    }
    
    private JPanel createOutputPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createLineBorder(new Color(155, 89, 182)),
            " Analysis Results ",
            javax.swing.border.TitledBorder.DEFAULT_JUSTIFICATION,
            javax.swing.border.TitledBorder.DEFAULT_POSITION,
            new Font("Segoe UI", Font.BOLD, 12),
            new Color(155, 89, 182)
        ));
        
        // Toolbar
        JPanel toolbar = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 5));
        
        processButton = createButton("⚡ Process", new Color(46, 204, 113));
        processButton.addActionListener(e -> processText());
        processButton.setEnabled(false);
        toolbar.add(processButton);
        
        copyButton = createButton("📄 Copy", new Color(52, 152, 219));
        copyButton.addActionListener(e -> copyToClipboard());
        toolbar.add(copyButton);
        
        saveButton = createButton("💾 Save", new Color(52, 152, 219));
        saveButton.addActionListener(e -> saveToFile());
        toolbar.add(saveButton);
        
        panel.add(toolbar, BorderLayout.NORTH);
        
        // Output Area
        outputTextArea = new JTextArea();
        outputTextArea.setFont(new Font("Consolas", Font.PLAIN, 13));
        outputTextArea.setLineWrap(false);
        outputTextArea.setEditable(false);
        outputTextArea.setBorder(new EmptyBorder(10, 10, 10, 10));
        outputTextArea.setBackground(new Color(255, 253, 240));
        
        JScrollPane scrollPane = new JScrollPane(outputTextArea);
        scrollPane.setBorder(BorderFactory.createLineBorder(new Color(200, 200, 200)));
        panel.add(scrollPane, BorderLayout.CENTER);
        
        return panel;
    }
    
    private JButton createButton(String text, Color color) {
        JButton button = new JButton(text);
        button.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        button.setBackground(color);
        button.setForeground(Color.WHITE);
        button.setFocusPainted(false);
        button.setBorderPainted(false);
        button.setOpaque(true);
        button.setPreferredSize(new Dimension(90, 30));
        
        button.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseEntered(java.awt.MouseEvent evt) {
                button.setBackground(color.darker());
            }
            
            public void mouseExited(java.awt.MouseEvent evt) {
                button.setBackground(color);
            }
        });
        
        return button;
    }
    
    private JPanel createStatusPanel() {
        JPanel panel = new JPanel(new BorderLayout(10, 5));
        panel.setBorder(new EmptyBorder(10, 0, 0, 0));
        
        statusLabel = new JLabel("⏳ Loading pipeline...");
        statusLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        statusLabel.setForeground(new Color(243, 156, 18));
        panel.add(statusLabel, BorderLayout.WEST);
        
        progressBar = new JProgressBar();
        progressBar.setIndeterminate(true);
        progressBar.setVisible(false);
        panel.add(progressBar, BorderLayout.CENTER);
        
        statsLabel = new JLabel("");
        statsLabel.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        statsLabel.setForeground(Color.GRAY);
        statsLabel.setHorizontalAlignment(SwingConstants.RIGHT);
        panel.add(statsLabel, BorderLayout.EAST);
        
        return panel;
    }
    
    private void processText() {
        if (isProcessing) return;
        
        String input = inputTextArea.getText().trim();
        if (input.isEmpty()) {
            JOptionPane.showMessageDialog(this, 
                "Please enter some text to analyze.", 
                "Empty Input", 
                JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        isProcessing = true;
        processButton.setEnabled(false);
        progressBar.setVisible(true);
        statusLabel.setText("⏳ Processing...");
        statusLabel.setForeground(new Color(243, 156, 18));
        outputTextArea.setText("");
        
        SwingWorker<Annotation, Void> worker = new SwingWorker<Annotation, Void>() {
            @Override
            protected Annotation doInBackground() throws Exception {
                List<String> annotatorsList = new ArrayList<>();
                if (wsegCheckBox.isSelected()) annotatorsList.add("wseg");
                if (posCheckBox.isSelected()) annotatorsList.add("pos");
                if (nerCheckBox.isSelected()) annotatorsList.add("ner");
                if (parseCheckBox.isSelected()) annotatorsList.add("parse");
                
                String[] annotators = annotatorsList.toArray(new String[0]);
                VnCoreNLP tempPipeline = new VnCoreNLP(annotators);
                
                Annotation annotation = new Annotation(input);
                tempPipeline.annotate(annotation);
                
                return annotation;
            }
            
            @Override
            protected void done() {
                try {
                    Annotation annotation = get();
                    String result = formatOutput(annotation);
                    outputTextArea.setText(result);
                    
                    // Update stats
                    int sentences = annotation.getSentences().size();
                    int words = 0;
                    for (Sentence sentence : annotation.getSentences()) {
                        words += sentence.getWords().size();
                    }
                    statsLabel.setText(String.format("📊 Sentences: %d | Words: %d", sentences, words));
                    
                    statusLabel.setText("✓ Processing completed");
                    statusLabel.setForeground(new Color(0, 128, 0));
                } catch (Exception e) {
                    statusLabel.setText("✗ Error: " + e.getMessage());
                    statusLabel.setForeground(Color.RED);
                    outputTextArea.setText("Error processing text:\n" + e.getMessage());
                    e.printStackTrace();
                } finally {
                    isProcessing = false;
                    processButton.setEnabled(true);
                    progressBar.setVisible(false);
                }
            }
        };
        
        worker.execute();
    }
    
    private String formatOutput(Annotation annotation) {
        String format = (String) outputFormatCombo.getSelectedItem();
        
        switch (format) {
            case "JSON":
                return formatAsJSON(annotation);
            case "Plain Text":
                return formatAsPlainText(annotation);
            case "HTML":
                return formatAsHTML(annotation);
            default:
                return annotation.toString();
        }
    }
    
    private String formatAsJSON(Annotation annotation) {
        StringBuilder json = new StringBuilder();
        json.append("{\n  \"sentences\": [\n");
        
        List<Sentence> sentences = annotation.getSentences();
        for (int i = 0; i < sentences.size(); i++) {
            Sentence sentence = sentences.get(i);
            json.append("    {\n");
            json.append("      \"id\": ").append(i + 1).append(",\n");
            json.append("      \"words\": [\n");
            
            List<Word> words = sentence.getWords();
            for (int j = 0; j < words.size(); j++) {
                Word word = words.get(j);
                json.append("        {\n");
                json.append("          \"text\": \"").append(escapeJson(word.getForm())).append("\",\n");
                json.append("          \"pos\": \"").append(word.getPos()).append("\",\n");
                json.append("          \"ner\": \"").append(word.getNer()).append("\",\n");
                json.append("          \"head\": ").append(word.getHead()).append(",\n");
                json.append("          \"dep\": \"").append(word.getDep()).append("\"\n");
                json.append("        }");
                if (j < words.size() - 1) json.append(",");
                json.append("\n");
            }
            
            json.append("      ]\n");
            json.append("    }");
            if (i < sentences.size() - 1) json.append(",");
            json.append("\n");
        }
        
        json.append("  ]\n}");
        return json.toString();
    }
    
    private String formatAsPlainText(Annotation annotation) {
        StringBuilder sb = new StringBuilder();
        
        for (Sentence sentence : annotation.getSentences()) {
            for (Word word : sentence.getWords()) {
                sb.append(word.getForm())
                  .append("/")
                  .append(word.getPos())
                  .append("/")
                  .append(word.getNer())
                  .append("  ");
            }
            sb.append("\n\n");
        }
        
        return sb.toString();
    }
    
    private String formatAsHTML(Annotation annotation) {
        StringBuilder html = new StringBuilder();
        html.append("<html><head><style>")
            .append("body { font-family: Arial, sans-serif; margin: 20px; }")
            .append(".sentence { margin-bottom: 20px; padding: 10px; background: #f5f5f5; border-radius: 5px; }")
            .append(".word { display: inline-block; margin: 2px; padding: 4px 8px; background: #3498db; color: white; border-radius: 3px; }")
            .append(".ner-PER { background: #e74c3c !important; }")
            .append(".ner-LOC { background: #2ecc71 !important; }")
            .append(".ner-ORG { background: #f39c12 !important; }")
            .append("</style></head><body>");
        
        for (Sentence sentence : annotation.getSentences()) {
            html.append("<div class='sentence'>");
            for (Word word : sentence.getWords()) {
                String nerClass = word.getNer().equals("O") ? "" : " ner-" + word.getNer().replace("-", "");
                html.append("<span class='word").append(nerClass).append("' title='")
                    .append("POS: ").append(word.getPos())
                    .append(", NER: ").append(word.getNer())
                    .append(", Dep: ").append(word.getDep())
                    .append("'>")
                    .append(word.getForm())
                    .append("</span> ");
            }
            html.append("</div>");
        }
        
        html.append("</body></html>");
        return html.toString();
    }
    
    private String escapeJson(String text) {
        return text.replace("\\", "\\\\")
                   .replace("\"", "\\\"")
                   .replace("\n", "\\n")
                   .replace("\r", "\\r")
                   .replace("\t", "\\t");
    }
    
    private void loadFile() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Open Text File");
        fileChooser.setFileFilter(new FileNameExtensionFilter("Text Files", "txt", "utf8", "vn"));
        fileChooser.setAcceptAllFileFilterUsed(false);
        
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            try {
                File file = fileChooser.getSelectedFile();
                StringBuilder content = new StringBuilder();
                try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        content.append(line).append("\n");
                    }
                }
                inputTextArea.setText(content.toString());
                statusLabel.setText("✓ File loaded: " + file.getName());
                statusLabel.setForeground(new Color(0, 128, 0));
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, 
                    "Error loading file: " + e.getMessage(), 
                    "Error", 
                    JOptionPane.ERROR_MESSAGE);
            }
        }
    }
    
    private void pasteFromClipboard() {
        try {
            Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
            String text = (String) clipboard.getData(DataFlavor.stringFlavor);
            inputTextArea.setText(text);
            statusLabel.setText("✓ Pasted from clipboard");
            statusLabel.setForeground(new Color(0, 128, 0));
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, 
                "Unable to paste from clipboard", 
                "Error", 
                JOptionPane.ERROR_MESSAGE);
        }
    }
    
    private void copyToClipboard() {
        if (outputTextArea.getText().isEmpty()) {
            JOptionPane.showMessageDialog(this, 
                "No results to copy", 
                "Empty Output", 
                JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        StringSelection selection = new StringSelection(outputTextArea.getText());
        Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
        clipboard.setContents(selection, selection);
        
        statusLabel.setText("✓ Copied to clipboard");
        statusLabel.setForeground(new Color(0, 128, 0));
    }
    
    private void saveToFile() {
        if (outputTextArea.getText().isEmpty()) {
            JOptionPane.showMessageDialog(this, 
                "No results to save", 
                "Empty Output", 
                JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Save Results");
        
        String format = (String) outputFormatCombo.getSelectedItem();
        String extension = ".txt";
        if ("JSON".equals(format)) extension = ".json";
        else if ("HTML".equals(format)) extension = ".html";
        
        fileChooser.setSelectedFile(new File("vncorenlp_results" + extension));
        
        int result = fileChooser.showSaveDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            try {
                File file = fileChooser.getSelectedFile();
                try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
                    writer.write(outputTextArea.getText());
                }
                statusLabel.setText("✓ Saved to: " + file.getName());
                statusLabel.setForeground(new Color(0, 128, 0));
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, 
                    "Error saving file: " + e.getMessage(), 
                    "Error", 
                    JOptionPane.ERROR_MESSAGE);
            }
        }
    }
    
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
            } catch (Exception e) {
                e.printStackTrace();
            }
            
            VnCoreNLPApp app = new VnCoreNLPApp();
            app.setVisible(true);
        });
    }
}
