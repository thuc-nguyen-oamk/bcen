# VnCoreNLP Python Batch Processor

A high-performance command-line tool for processing large volumes of text files using the VnCoreNLP Vietnamese NLP library.

## Features

✅ **Word Segmentation** (wseg)  
✅ **POS Tagging** (pos)  
✅ **Named Entity Recognition** (ner)  
✅ **Dependency Parsing** (parse)  

### Key Features

- 🚀 Parallel processing with configurable thread count
- 📁 Recursive directory scanning for text files
- 📄 Multiple output formats: Tabular, JSON, CoNLL
- 📊 Progress reporting with success/failure counts
- ⚙️ Flexible annotation configuration
- 🔧 Command-line argument parsing

## Requirements

- Python 3.8 or higher
- Java 11 or higher
- pyjnius

## Installation

```bash
pip install pyjnius
```

## Usage

### Basic Usage

```bash
cd VnCoreNLP-Python-Batch
python batch_processor.py -i ./input -o ./output
```

### With Custom Options

```bash
python batch_processor.py \
    --input ./documents \
    --output ./results \
    --format json \
    --threads 8
```

### Disable Specific Annotations

```bash
python batch_processor.py \
    -i ./input -o ./output \
    --disable-parse --disable-ner
```

## Command-Line Options

```
-i, --input <DIR>         Input directory containing text files
-o, --output <DIR>        Output directory for processed results
-f, --format <FORMAT>     Output format: tabular, json, conll (default: tabular)
-t, --threads <NUM>       Number of parallel threads (default: 4)
    --disable-wseg        Disable word segmentation
    --disable-pos         Disable POS tagging
    --disable-ner         Disable NER
    --disable-parse       Disable dependency parsing
-h, --help                Print help message
```

## Output Formats

### Tabular Format (Default)
Human-readable format with structured tables showing tokens, POS tags, NER labels, and dependencies.

### JSON Format
Machine-readable JSON format with full annotation data including:
- Original text
- Sentences with tokens
- Lemmas, POS tags, NER labels
- Dependency parse trees
- Named entities

### CoNLL Format
Standard CoNLL-U format for compatibility with other NLP tools.

## Example

### Input File (input/sample.txt)
```
Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.
Bà Lan, vợ ông Chúc, cũng làm việc tại đây.
```

### Output File (output/sample_processed.txt)
```
================================================================================
VN CORE NLP BATCH PROCESSING RESULTS
================================================================================
Source: input/sample.txt
Processed at: 2024-01-15T10:30:45.123456

────────────────────────────────────────────────────────────────────────────────
SENTENCE 1: Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.
────────────────────────────────────────────────────────────────────────────────

Tokens:
No.   Word                 Lemma           POS        NER            
---------------------------------------------------------------------
1     Ông                  ông             N          O              
2     Nguyễn               nguyễn          N          P_PER          
3     Khắc                 khắc            N          P_PER          
4     Chúc                 chúc            N          P_PER          
...

Dependencies:
Gov    Dep    Relation            
-------------------------------------
2      1      nsubj               
4      2      flat                
4      3      flat                

════════════════════════════════════════════════════════════════════════════════
NAMED ENTITIES
════════════════════════════════════════════════════════════════════════════════
• Nguyễn Khắc Chúc [P_PER]
• Đại học Quốc gia Hà Nội [ORG]
```

## Performance Tips

1. **Use multiple threads**: For large datasets, increase thread count
   ```bash
   python batch_processor.py -i ./input -o ./output --threads 16
   ```

2. **Disable unused annotators**: Only enable what you need
   ```bash
   python batch_processor.py -i ./input -o ./output --disable-parse
   ```

3. **Use JSON for machine processing**: Faster to parse programmatically
   ```bash
   python batch_processor.py -i ./input -o ./output --format json
   ```

4. **Increase Java heap size**: For very large files, modify the script:
   ```python
   jnius_config.add_options('-Xmx8g', '-Xms2g')
   ```

## File Extensions

The processor automatically finds files with these extensions:
- `.txt`
- `.text`
- `.vi`
- `.utf8`

## Troubleshooting

### "VnCoreNLP not initialized" Error
- Ensure Java 11+ is installed
- Verify `JAVA_HOME` environment variable is set
- Check that `VnCoreNLP-1.2.jar` exists in `__VnCoreNLP/`

### Out of Memory Error
Increase Java heap size in the script:
```python
jnius_config.add_options('-Xmx8g', '-Xms2g')
```

### Slow Processing
- Reduce number of threads if CPU is overloaded
- Disable unused annotations
- Process files in smaller batches

## License

This application is built on top of the VnCoreNLP library. Please refer to the original VnCoreNLP license for usage terms.

## Credits

- **VnCoreNLP Library**: Original Vietnamese NLP library
- **pyjnius**: Python-Java bridge library

---

**Version**: 1.0.0  
**Last Updated**: 2024
