#!/usr/bin/env python3
"""
VnCoreNLP Command-Line Batch Processor

A high-performance command-line tool for processing large volumes of text files
using the VnCoreNLP Vietnamese NLP library.

Features:
- Word Segmentation (wseg)
- POS Tagging (pos)
- Named Entity Recognition (ner)
- Dependency Parsing (parse)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure JAVA_HOME and CLASSPATH
JAVA_HOME = os.environ.get('JAVA_HOME', '/usr/lib/jvm/java-17-openjdk-amd64')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VN_CORE_NLP_JAR = os.path.join(os.path.dirname(SCRIPT_DIR), '__VnCoreNLP', 'VnCoreNLP-1.2.jar')

os.environ['JAVA_HOME'] = JAVA_HOME

import jnius_config
jnius_config.add_options('-Xmx4g', '-Xms1g')
jnius_config.set_classpath(VN_CORE_NLP_JAR)

from jnius import autoclass


class VnCoreNLPBatchProcessor:
    """Batch processor for VnCoreNLP."""
    
    def __init__(self, annotators: Optional[List[str]] = None):
        self.annotator = None
        self.annotators = annotators or ['wseg', 'pos', 'ner', 'parse']
        self._initialize()
    
    def _initialize(self):
        """Initialize the VnCoreNLP annotator."""
        try:
            Properties = autoclass('java.util.Properties')
            Pipeline = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
            
            props = Properties()
            props.setProperty('annotators', ','.join(self.annotators))
            props.setProperty('ssplit.boundaryTokenRegex', r'[.।!?;:]+')
            props.setProperty('tokenize.language', 'vi')
            
            self.annotator = Pipeline(props)
            print(f"✓ VnCoreNLP initialized with annotators: {', '.join(self.annotators)}")
        except Exception as e:
            print(f"✗ Error initializing VnCoreNLP: {e}")
            raise
    
    def process_file(self, input_path: str, output_dir: str, output_format: str = 'tabular') -> Dict[str, Any]:
        """Process a single file and save results."""
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Process text
            result = self._process_text(text)
            
            # Add metadata
            result['source_file'] = input_path
            result['processed_at'] = datetime.now().isoformat()
            
            # Create output filename
            input_name = Path(input_path).stem
            output_ext = {'tabular': 'txt', 'json': 'json', 'conll': 'conll'}.get(output_format, 'txt')
            output_path = os.path.join(output_dir, f"{input_name}_processed.{output_ext}")
            
            # Format and save output
            formatted = self._format_output(result, output_format)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            
            return {
                'success': True,
                'input': input_path,
                'output': output_path,
                'sentences': len(result['sentences']),
                'tokens': sum(len(s['tokens']) for s in result['sentences'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'input': input_path,
                'error': str(e)
            }
    
    def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text and extract annotations."""
        Annotation = autoclass('edu.stanford.nlp.pipeline.Annotation')
        CoreAnnotations = autoclass('edu.stanford.nlp.ling.CoreAnnotations')
        
        annotation = Annotation(text)
        self.annotator.annotate(annotation)
        
        result = {
            'text': text,
            'sentences': []
        }
        
        sentences = annotation.get(CoreAnnotations.SentencesAnnotation).class_
        
        for i in range(sentences.size()):
            sentence_data = self._extract_sentence_data(sentences.get(i))
            result['sentences'].append(sentence_data)
        
        result['entities'] = self._extract_entities(annotation, CoreAnnotations)
        
        return result
    
    def _extract_sentence_data(self, sentence) -> Dict[str, Any]:
        """Extract data from a single sentence."""
        CoreAnnotations = autoclass('edu.stanford.nlp.ling.CoreAnnotations')
        
        sentence_data = {
            'text': sentence.get(CoreAnnotations.TextAnnotation).class_,
            'tokens': [],
            'dependencies': []
        }
        
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
    
    def _extract_entities(self, annotation, CoreAnnotations) -> List[Dict[str, Any]]:
        """Extract named entities."""
        entities = []
        try:
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
                            else:
                                if current_entity:
                                    current_entity['end'] = j - 1
                                    entities.append(current_entity)
                                    current_entity = None
                        
                        if current_entity:
                            current_entity['end'] = tokens.size() - 1
                            entities.append(current_entity)
                except:
                    break
        except Exception as e:
            print(f"Warning: Error extracting entities: {e}")
        
        return entities
    
    def _format_output(self, result: Dict, format_type: str) -> str:
        """Format output according to specified type."""
        if format_type == 'json':
            return json.dumps(result, indent=2, ensure_ascii=False)
        
        elif format_type == 'conll':
            lines = []
            for sent_idx, sent in enumerate(result['sentences'], 1):
                lines.append(f"# sentence {sent_idx}")
                lines.append(f"# text = {sent['text']}")
                
                for token in sent['tokens']:
                    deps = "_"
                    line = (
                        f"{token['index']}\t"
                        f"{token['word']}\t"
                        f"{token['lemma']}\t"
                        f"{token['pos']}\t"
                        f"{token['pos']}\t"
                        f"_\t"
                        f"{deps}\t"
                        f"{deps}\t"
                        f"{token['ner']}\t"
                        f"_"
                    )
                    lines.append(line)
                lines.append("")
            
            return "\n".join(lines)
        
        else:  # tabular
            lines = []
            lines.append("=" * 80)
            lines.append("VN CORE NLP BATCH PROCESSING RESULTS")
            lines.append("=" * 80)
            lines.append(f"Source: {result.get('source_file', 'Unknown')}")
            lines.append(f"Processed at: {result.get('processed_at', 'Unknown')}")
            lines.append("")
            
            for i, sent in enumerate(result['sentences'], 1):
                lines.append(f"\n{'─' * 80}")
                lines.append(f"SENTENCE {i}: {sent['text']}")
                lines.append(f"{'─' * 80}")
                
                if sent['tokens']:
                    lines.append("\nTokens:")
                    lines.append(f"{'No.':<5} {'Word':<20} {'Lemma':<15} {'POS':<10} {'NER':<15}")
                    lines.append("-" * 65)
                    
                    for token in sent['tokens']:
                        lines.append(
                            f"{token['index']:<5} "
                            f"{token['word']:<20} "
                            f"{token['lemma']:<15} "
                            f"{token['pos']:<10} "
                            f"{token['ner']:<15}"
                        )
                
                if sent['dependencies']:
                    lines.append("\nDependencies:")
                    lines.append(f"{'Gov':<6} {'Dep':<6} {'Relation':<20}")
                    lines.append("-" * 32)
                    
                    for dep in sent['dependencies']:
                        lines.append(
                            f"{dep['governor']:<6} "
                            f"{dep['dependent']:<6} "
                            f"{dep['relation']:<20}"
                        )
            
            if result['entities']:
                lines.append(f"\n{'═' * 80}")
                lines.append("NAMED ENTITIES")
                lines.append(f"{'═' * 80}")
                for entity in result['entities']:
                    lines.append(f"• {entity['text']} [{entity['type']}]")
            
            return "\n".join(lines)


def find_text_files(input_dir: str) -> List[str]:
    """Recursively find all text files in directory."""
    text_files = []
    extensions = {'.txt', '.text', '.vi', '.utf8'}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                text_files.append(os.path.join(root, file))
    
    return text_files


def main():
    parser = argparse.ArgumentParser(
        description='VnCoreNLP Batch Processor - Process multiple text files with Vietnamese NLP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i ./input -o ./output
  %(prog)s --input ./documents --output ./results --format json --threads 8
  %(prog)s -i ./input -o ./output --disable-parse --disable-ner
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input directory containing text files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for processed results')
    parser.add_argument('-f', '--format', choices=['tabular', 'json', 'conll'], default='tabular',
                       help='Output format (default: tabular)')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Number of parallel threads (default: 4)')
    
    parser.add_argument('--disable-wseg', action='store_true', help='Disable word segmentation')
    parser.add_argument('--disable-pos', action='store_true', help='Disable POS tagging')
    parser.add_argument('--disable-ner', action='store_true', help='Disable NER')
    parser.add_argument('--disable-parse', action='store_true', help='Disable dependency parsing')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine enabled annotators
    annotators = []
    if not args.disable_wseg:
        annotators.append('wseg')
    if not args.disable_pos:
        annotators.append('pos')
    if not args.disable_ner:
        annotators.append('ner')
    if not args.disable_parse:
        annotators.append('parse')
    
    if not annotators:
        print("Error: At least one annotator must be enabled!")
        sys.exit(1)
    
    print(f"VnCoreNLP Batch Processor")
    print(f"=" * 50)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Output format: {args.format}")
    print(f"Threads: {args.threads}")
    print(f"Enabled annotators: {', '.join(annotators)}")
    print(f"=" * 50)
    
    # Find text files
    text_files = find_text_files(args.input)
    if not text_files:
        print(f"No text files found in {args.input}")
        sys.exit(0)
    
    print(f"Found {len(text_files)} text files to process")
    print()
    
    # Initialize processor
    try:
        processor = VnCoreNLPBatchProcessor(annotators)
    except Exception as e:
        print(f"Failed to initialize processor: {e}")
        sys.exit(1)
    
    # Process files
    success_count = 0
    failure_count = 0
    total_tokens = 0
    total_sentences = 0
    
    progress_lock = threading.Lock()
    progress_counter = [0]
    
    def process_with_progress(file_path):
        result = processor.process_file(file_path, args.output, args.format)
        
        with progress_lock:
            progress_counter[0] += 1
            status = "✓" if result['success'] else "✗"
            print(f"[{progress_counter[0]}/{len(text_files)}] {status} {os.path.basename(file_path)}")
            
            if result['success']:
                global success_count, total_tokens, total_sentences
                success_count += 1
                total_tokens += result.get('tokens', 0)
                total_sentences += result.get('sentences', 0)
            else:
                global failure_count
                failure_count += 1
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        return result
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_with_progress, f) for f in text_files]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Unexpected error: {e}")
    
    # Print summary
    print()
    print("=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total files:     {len(text_files)}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {failure_count}")
    print(f"Total sentences: {total_sentences}")
    print(f"Total tokens:    {total_tokens}")
    print(f"Output format:   {args.format}")
    print(f"Output dir:      {args.output}")
    print("=" * 50)
    
    if failure_count > 0:
        print(f"\n⚠ Warning: {failure_count} file(s) failed to process")
    
    print("\n✓ Batch processing completed!")


if __name__ == "__main__":
    main()
