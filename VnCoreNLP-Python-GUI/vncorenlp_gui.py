#!/usr/bin/env python3
"""
VnCoreNLP Python GUI Application
A modern tkinter-based desktop application for Vietnamese NLP processing.

Features:
- Word Segmentation (wseg)
- POS Tagging (pos)
- Named Entity Recognition (ner)
- Dependency Parsing (parse)

This application uses pyjnius to interface with the VnCoreNLP Java library.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, Menu
import json
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any

# Configure JAVA_HOME and CLASSPATH before importing jnius
JAVA_HOME = os.environ.get('JAVA_HOME', '/usr/lib/jvm/java-17-openjdk-amd64')
VN_CORE_NLP_JAR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                '__VnCoreNLP', 'VnCoreNLP-1.2.jar')

os.environ['JAVA_HOME'] = JAVA_HOME

import jnius_config
jnius_config.add_options('-Xmx4g', '-Xms1g')
jnius_config.set_classpath(VN_CORE_NLP_JAR)

from jnius import autoclass, JavaException


class VnCoreNLPProcessor:
    """Python wrapper for VnCoreNLP Java library."""
    
    def __init__(self):
        self.annotator = None
        self.is_initialized = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the VnCoreNLP annotator."""
        try:
            # Import Java classes
            Properties = autoclass('java.util.Properties')
            Pipeline = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
            
            # Configure properties for Vietnamese NLP
            props = Properties()
            props.setProperty('annotators', 'wseg,pos,ner,parse')
            props.setProperty('ssplit.boundaryTokenRegex', r'[.।!?;:]+')
            props.setProperty('tokenize.language', 'vi')
            
            # Create annotator
            self.annotator = Pipeline(props)
            self.is_initialized = True
            print("✓ VnCoreNLP initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing VnCoreNLP: {e}")
            self.is_initialized = False
    
    def process_text(self, text: str, annotators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process Vietnamese text with specified annotations.
        
        Args:
            text: Input Vietnamese text
            annotators: List of annotators to use ['wseg', 'pos', 'ner', 'parse']
        
        Returns:
            Dictionary containing processed results
        """
        if not self.is_initialized:
            raise RuntimeError("VnCoreNLP not initialized")
        
        try:
            # Create annotation
            Annotation = autoclass('edu.stanford.nlp.pipeline.Annotation')
            annotation = Annotation(text)
            
            # Run pipeline
            self.annotator.annotate(annotation)
            
            # Extract results
            result = {
                'text': text,
                'timestamp': datetime.now().isoformat(),
                'sentences': []
            }
            
            # Get sentences
            CoreAnnotations = autoclass('edu.stanford.nlp.ling.CoreAnnotations')
            sentences = annotation.get(CoreAnnotations.SentencesAnnotation).class_
            
            for i in range(sentences.size()):
                sentence_data = self._extract_sentence_data(sentences.get(i))
                result['sentences'].append(sentence_data)
            
            # Extract entities
            result['entities'] = self._extract_entities(annotation)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error processing text: {e}")
    
    def _extract_sentence_data(self, sentence) -> Dict[str, Any]:
        """Extract data from a single sentence."""
        try:
            CoreAnnotations = autoclass('edu.stanford.nlp.ling.CoreAnnotations')
            
            sentence_data = {
                'text': sentence.get(CoreAnnotations.TextAnnotation).class_,
                'tokens': [],
                'dependencies': []
            }
            
            # Extract tokens
            tokens = sentence.get(CoreAnnotations.TokensAnnotation).class_
            for i in range(tokens.size()):
                token = tokens.get(i)
                token_data = {
                    'word': token.get(CoreAnnotations.TextAnnotation).class_,
                    'lemma': token.get(CoreAnnotations.LemmaAnnotation).class_,
                    'pos': token.get(CoreAnnotations.PartOfSpeechAnnotation).class_,
                    'ner': token.get(CoreAnnotations.NamedEntityTagAnnotation).class_,
                    'index': i + 1
                }
                sentence_data['tokens'].append(token_data)
            
            # Extract dependencies
            try:
                dependencies = sentence.get(CoreAnnotations.BasicDependenciesAnnotation).class_
                for dep in dependencies.toList():
                    dep_data = {
                        'governor': dep.getGovernerIndex(),
                        'dependent': dep.getDependentIndex(),
                        'relation': dep.getDepLabelString()
                    }
                    sentence_data['dependencies'].append(dep_data)
            except:
                pass
            
            return sentence_data
            
        except Exception as e:
            print(f"Error extracting sentence data: {e}")
            return {'text': '', 'tokens': [], 'dependencies': []}
    
    def _extract_entities(self, annotation) -> List[Dict[str, Any]]:
        """Extract named entities from annotation."""
        entities = []
        try:
            CoreAnnotations = autoclass('edu.stanford.nlp.ling.CoreAnnotations')
            
            # Try to get entity mentions
            sentences = annotation.get(CoreAnnotations.SentencesAnnotation).class_
            for i in range(1, sentences.size() + 1):
                try:
                    sentence = sentences.get(i)
                    if sentence:
                        tokens = sentence.get(CoreAnnotations.TokensAnnotation).class_
                        current_entity = None
                        
                        for j in range(tokens.size()):
                            token = tokens.get(j)
                            ner_tag = token.get(CoreAnnotations.NamedEntityTagAnnotation).class_
                            
                            if ner_tag and ner_tag != 'O':
                                if current_entity is None:
                                    current_entity = {
                                        'text': token.get(CoreAnnotations.TextAnnotation).class_,
                                        'type': ner_tag,
                                        'start': j
                                    }
                                else:
                                    current_entity['text'] += ' ' + token.get(CoreAnnotations.TextAnnotation).class_
                                    current_entity['end'] = j
                            else:
                                if current_entity:
                                    if 'end' not in current_entity:
                                        current_entity['end'] = j - 1
                                    entities.append(current_entity)
                                    current_entity = None
                        
                        if current_entity:
                            if 'end' not in current_entity:
                                current_entity['end'] = tokens.size() - 1
                            entities.append(current_entity)
                except:
                    break
        except Exception as e:
            print(f"Error extracting entities: {e}")
        
        return entities
    
    def segment_words(self, text: str) -> str:
        """Perform word segmentation only."""
        result = self.process_text(text, ['wseg'])
        segmented = []
        for sent in result['sentences']:
            words = [t['word'] for t in sent['tokens']]
            segmented.append(' '.join(words))
        return ' | '.join(segmented)
    
    def get_pos_tags(self, text: str) -> List[tuple]:
        """Get POS tags for text."""
        result = self.process_text(text, ['wseg', 'pos'])
        tags = []
        for sent in result['sentences']:
            for token in sent['tokens']:
                tags.append((token['word'], token['pos']))
        return tags
    
    def get_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities."""
        result = self.process_text(text, ['wseg', 'pos', 'ner'])
        return result['entities']


class VnCoreNLPGUI:
    """Main GUI Application class."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VnCoreNLP - Vietnamese NLP Processor")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'bg': '#f0f0f0',
            'primary': '#2196F3',
            'secondary': '#4CAF50',
            'accent': '#FF5722',
            'text': '#333333',
            'light': '#ffffff',
            'dark': '#2c3e50'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Initialize processor
        self.processor: Optional[VnCoreNLPProcessor] = None
        self.current_result: Optional[Dict] = None
        
        # Setup UI
        self._setup_menu()
        self._setup_ui()
        
        # Initialize processor in background
        self._init_processor_async()
    
    def _setup_menu(self):
        """Setup application menu."""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Text File...", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Results...", command=self._save_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Edit menu
        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Results", command=self._copy_results, accelerator="Ctrl+C")
        edit_menu.add_command(label="Clear All", command=self._clear_all, accelerator="Ctrl+L")
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        
        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._open_file())
        self.root.bind('<Control-s>', lambda e: self._save_results())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<Control-l>', lambda e: self._clear_all())
    
    def _setup_ui(self):
        """Setup the main UI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(
            header_frame,
            text="🇻🇳 VnCoreNLP - Vietnamese NLP Processor",
            font=('Helvetica', 18, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(
            header_frame,
            text="Initializing...",
            font=('Helvetica', 10),
            foreground='gray'
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Annotation options
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill=tk.X)
        
        self.wseg_var = tk.BooleanVar(value=True)
        self.pos_var = tk.BooleanVar(value=True)
        self.ner_var = tk.BooleanVar(value=True)
        self.parse_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="Word Segmentation", variable=self.wseg_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(options_frame, text="POS Tagging", variable=self.pos_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(options_frame, text="Named Entity Recognition", variable=self.ner_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(options_frame, text="Dependency Parsing", variable=self.parse_var).pack(side=tk.LEFT, padx=5)
        
        # Output format
        format_frame = ttk.Frame(control_frame)
        format_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(format_frame, text="Output Format:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.format_var = tk.StringVar(value="tabular")
        format_combo = ttk.Combobox(format_frame, textvariable=self.format_var, width=15, state="readonly")
        format_combo['values'] = ('tabular', 'json', 'plain', 'html')
        format_combo.pack(side=tk.LEFT)
        
        # Process button
        self.process_btn = ttk.Button(
            control_frame,
            text="⚡ Process Text",
            command=self._process_text,
            state='disabled'
        )
        self.process_btn.pack(side=tk.RIGHT, padx=5)
        
        # Clear button
        clear_btn = ttk.Button(
            control_frame,
            text="🗑️ Clear",
            command=self._clear_all
        )
        clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # Content area (split view)
        content_frame = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Input panel
        input_frame = ttk.LabelFrame(content_frame, text="Input Text", padding="10")
        content_frame.add(input_frame, weight=1)
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            bg='#fafafa',
            padx=10,
            pady=10
        )
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # Sample text button
        sample_btn = ttk.Button(
            input_frame,
            text="📝 Load Sample Text",
            command=self._load_sample_text
        )
        sample_btn.pack(pady=(5, 0))
        
        # Output panel
        output_frame = ttk.LabelFrame(content_frame, text="Results", padding="10")
        content_frame.add(output_frame, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#fafafa',
            padx=10,
            pady=10,
            state='disabled'
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Statistics frame
        stats_frame = ttk.Frame(main_frame)
        stats_frame.pack(fill=tk.X)
        
        self.stats_label = ttk.Label(
            stats_frame,
            text="",
            font=('Helvetica', 9),
            foreground='gray'
        )
        self.stats_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(side=tk.RIGHT)
    
    def _init_processor_async(self):
        """Initialize VnCoreNLP processor in background thread."""
        def init():
            try:
                self.processor = VnCoreNLPProcessor()
                self.root.after(0, self._on_processor_ready)
            except Exception as e:
                self.root.after(0, lambda: self._on_processor_error(str(e)))
        
        import threading
        thread = threading.Thread(target=init, daemon=True)
        thread.start()
    
    def _on_processor_ready(self):
        """Callback when processor is ready."""
        self.status_label.config(text="✓ Ready", foreground='green')
        self.process_btn.config(state='normal')
        messagebox.showinfo(
            "VnCoreNLP Ready",
            "VnCoreNLP has been initialized successfully!\n\n"
            "You can now process Vietnamese text with:\n"
            "- Word Segmentation\n"
            "- POS Tagging\n\n"
            "- Named Entity Recognition\n"
            "- Dependency Parsing"
        )
    
    def _on_processor_error(self, error: str):
        """Callback when processor initialization fails."""
        self.status_label.config(text="✗ Error", foreground='red')
        messagebox.showerror(
            "Initialization Error",
            f"Failed to initialize VnCoreNLP:\n{error}\n\n"
            "Please ensure:\n"
            "1. Java 11+ is installed\n"
            "2. VnCoreNLP-1.2.jar exists in __VnCoreNLP/\n"
            "3. pyjnius is installed correctly"
        )
    
    def _load_sample_text(self):
        """Load sample Vietnamese text."""
        sample_text = """Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. 
Bà Lan, vợ ông Chúc, cũng làm việc tại đây.

Công ty TNHH MTV Việt Nam có trụ sở chính tại Hà Nội và chi nhánh tại Thành phố Hồ Chí Minh.

Chủ tịch nước Võ Văn Thưởng đã gặp gỡ các đại biểu trẻ tiêu biểu toàn quốc."""
        
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(tk.END, sample_text)
    
    def _process_text(self):
        """Process the input text."""
        if not self.processor:
            messagebox.showwarning("Warning", "Processor not initialized yet!")
            return
        
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to process!")
            return
        
        # Show progress
        self.progress.start()
        self.process_btn.config(state='disabled')
        
        def process():
            try:
                # Get selected annotators
                annotators = []
                if self.wseg_var.get():
                    annotators.append('wseg')
                if self.pos_var.get():
                    annotators.append('pos')
                if self.ner_var.get():
                    annotators.append('ner')
                if self.parse_var.get():
                    annotators.append('parse')
                
                # Process text
                result = self.processor.process_text(text, annotators)
                self.current_result = result
                
                # Format output
                output_format = self.format_var.get()
                formatted = self._format_output(result, output_format)
                
                # Update stats
                stats = self._calculate_stats(result)
                
                self.root.after(0, lambda: self._display_result(formatted, stats))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"))
            finally:
                self.root.after(0, self._processing_complete)
        
        import threading
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def _display_result(self, formatted: str, stats: str):
        """Display the result in output panel."""
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, formatted)
        self.output_text.config(state='disabled')
        self.stats_label.config(text=stats)
    
    def _processing_complete(self):
        """Called when processing is complete."""
        self.progress.stop()
        self.process_btn.config(state='normal')
    
    def _format_output(self, result: Dict, format_type: str) -> str:
        """Format the output according to specified type."""
        if format_type == 'json':
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        elif format_type == 'tabular':
            lines = []
            lines.append("=" * 80)
            lines.append("VN CORE NLP ANALYSIS RESULTS")
            lines.append("=" * 80)
            lines.append(f"Processed at: {result['timestamp']}")
            lines.append("")
            
            for i, sent in enumerate(result['sentences'], 1):
                lines.append(f"\n{'─' * 80}")
                lines.append(f"SENTENCE {i}: {sent['text']}")
                lines.append(f"{'─' * 80}")
                
                if sent['tokens']:
                    lines.append("\nTokens:")
                    lines.append(f"{'No.':<5} {'Word':<20} {'Lemma':<20} {'POS':<10} {'NER':<15}")
                    lines.append("-" * 70)
                    
                    for token in sent['tokens']:
                        lines.append(
                            f"{token['index']:<5} "
                            f"{token['word']:<20} "
                            f"{token['lemma']:<20} "
                            f"{token['pos']:<10} "
                            f"{token['ner']:<15}"
                        )
                
                if sent['dependencies']:
                    lines.append("\nDependencies:")
                    lines.append(f"{'Governor':<10} {'Dependent':<10} {'Relation':<20}")
                    lines.append("-" * 40)
                    
                    for dep in sent['dependencies']:
                        lines.append(
                            f"{dep['governor']:<10} "
                            f"{dep['dependent']:<10} "
                            f"{dep['relation']:<20}"
                        )
            
            if result['entities']:
                lines.append(f"\n{'═' * 80}")
                lines.append("NAMED ENTITIES")
                lines.append(f"{'═' * 80}")
                for entity in result['entities']:
                    lines.append(f"• {entity['text']} [{entity['type']}]")
            
            return "\n".join(lines)
        
        elif format_type == 'html':
            html = ["<!DOCTYPE html>", "<html><head>", 
                   "<style>",
                   "body { font-family: Arial, sans-serif; margin: 20px; }",
                   ".sentence { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }",
                   ".token { display: inline-block; margin: 3px; padding: 5px 10px; background: #e3f2fd; border-radius: 3px; }",
                   ".pos { color: #1976d2; font-weight: bold; }",
                   ".ner { color: #d32f2f; font-weight: bold; }",
                   ".entity { background: #ffcdd2 !important; }",
                   "</style>",
                   "</head><body>"]
            
            html.append(f"<h1>VnCoreNLP Analysis Results</h1>")
            html.append(f"<p>Processed at: {result['timestamp']}</p>")
            
            for i, sent in enumerate(result['sentences'], 1):
                html.append(f"<div class='sentence'>")
                html.append(f"<h3>Sentence {i}</h3>")
                html.append("<div>")
                
                for token in sent['tokens']:
                    css_class = "token"
                    if token['ner'] and token['ner'] != 'O':
                        css_class += " entity"
                    
                    html.append(
                        f"<span class='{css_class}'>"
                        f"{token['word']} "
                        f"<span class='pos'>[{token['pos']}]</span>"
                    )
                    if token['ner'] and token['ner'] != 'O':
                        html.append(f" <span class='ner'>[{token['ner']}]</span>")
                    html.append("</span>")
                
                html.append("</div></div>")
            
            html.append("</body></html>")
            return "\n".join(html)
        
        else:  # plain
            lines = []
            for sent in result['sentences']:
                words = [t['word'] for t in sent['tokens']]
                lines.append(' '.join(words))
            return '\n'.join(lines)
    
    def _calculate_stats(self, result: Dict) -> str:
        """Calculate and return statistics."""
        num_sentences = len(result['sentences'])
        num_tokens = sum(len(s['tokens']) for s in result['sentences'])
        num_entities = len(result.get('entities', []))
        
        return f"Sentences: {num_sentences} | Words: {num_tokens} | Entities: {num_entities}"
    
    def _open_file(self):
        """Open a text file."""
        filetypes = [
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Open Text File",
            filetypes=filetypes
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.input_text.delete(1.0, tk.END)
                self.input_text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file:\n{e}")
    
    def _save_results(self):
        """Save results to file."""
        if not self.current_result:
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        filetypes = [
            ("JSON files", "*.json"),
            ("Text files", "*.txt"),
            ("HTML files", "*.html"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            filetypes=filetypes,
            defaultextension=".txt"
        )
        
        if filename:
            try:
                format_type = filename.split('.')[-1].lower()
                if format_type == 'json':
                    content = json.dumps(self.current_result, indent=2, ensure_ascii=False)
                elif format_type == 'html':
                    content = self._format_output(self.current_result, 'html')
                else:
                    content = self._format_output(self.current_result, 'tabular')
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                messagebox.showinfo("Success", f"Results saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")
    
    def _copy_results(self):
        """Copy results to clipboard."""
        if not self.current_result:
            messagebox.showwarning("Warning", "No results to copy!")
            return
        
        content = self._format_output(self.current_result, self.format_var.get())
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        messagebox.showinfo("Success", "Results copied to clipboard!")
    
    def _clear_all(self):
        """Clear all text and results."""
        self.input_text.delete(1.0, tk.END)
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        self.stats_label.config(text="")
        self.current_result = None
    
    def _show_about(self):
        """Show about dialog."""
        about_text = """VnCoreNLP Python GUI Application
        
Version: 1.0.0

A modern desktop application for Vietnamese Natural Language Processing.

Features:
• Word Segmentation (wseg)
• POS Tagging (pos)
• Named Entity Recognition (ner)
• Dependency Parsing (parse)

Built with:
• Python 3 & tkinter
• pyjnius (Java-Python bridge)
• VnCoreNLP 1.2

© 2024 - Built on VnCoreNLP library"""
        
        messagebox.showinfo("About VnCoreNLP", about_text)


def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        # You can add an icon here
        pass
    except:
        pass
    
    app = VnCoreNLPGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
