# VnCoreNLP Python GUI Application

A modern tkinter-based desktop application for Vietnamese Natural Language Processing, built on top of the VnCoreNLP library.

## Features

✅ **Word Segmentation** (wseg) - Split Vietnamese text into words  
✅ **POS Tagging** (pos) - Assign part-of-speech tags to words  
✅ **Named Entity Recognition** (ner) - Identify named entities (persons, organizations, locations)  
✅ **Dependency Parsing** (parse) - Analyze grammatical structure and dependencies  

### Additional Features

- 🖥️ Modern, intuitive GUI with color-coded panels
- ⚡ Asynchronous processing with progress indicator
- 📊 Real-time statistics display (sentence count, word count, entity count)
- 📋 Copy/paste from clipboard support
- 💾 Save results in multiple formats (TXT, JSON, HTML)
- 📂 Load text files for analysis
- 🎨 Multiple output formats: Tabular, JSON, Plain Text, HTML
- ✅ Selectable annotation types
- ⌨️ Keyboard shortcuts for common operations

## Requirements

- Python 3.8 or higher
- Java 11 or higher
- pyjnius
- tkinter (usually included with Python)

## Installation

### 1. Install Python Dependencies

```bash
pip install pyjnius
```

### 2. Verify Java Installation

```bash
java -version
```

Make sure Java 11+ is installed and `JAVA_HOME` environment variable is set.

### 3. Verify VnCoreNLP JAR

Ensure that `VnCoreNLP-1.2.jar` exists in the `__VnCoreNLP/` directory at the root of this repository.

## Usage

### Running the Application

```bash
cd VnCoreNLP-Python-GUI
python vncorenlp_gui.py
```

Or directly:

```bash
python /workspace/VnCoreNLP-Python-GUI/vncorenlp_gui.py
```

### Basic Operations

1. **Load Sample Text**: Click "📝 Load Sample Text" to load example Vietnamese text
2. **Select Annotations**: Check/uncheck the annotation types you want to use
3. **Choose Output Format**: Select from tabular, JSON, plain, or HTML format
4. **Process**: Click "⚡ Process Text" to analyze the text
5. **View Results**: Results appear in the right panel with statistics
6. **Save/Export**: Use File menu to save results or copy to clipboard

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open text file |
| `Ctrl+S` | Save results |
| `Ctrl+Q` | Quit application |
| `Ctrl+L` | Clear all |
| `Ctrl+C` | Copy results |

## Output Formats

### Tabular Format
Displays results in a structured table with columns for:
- Token number
- Word
- Lemma
- POS tag
- Named entity tag
- Dependency relations

### JSON Format
Full structured data including:
- Original text
- Timestamp
- All sentences with tokens
- Named entities
- Dependency parse trees

### HTML Format
Beautiful, styled HTML output with:
- Color-coded tokens
- Highlighted named entities
- POS tags inline
- Responsive design

### Plain Format
Simple space-separated words per sentence.

## Sample Vietnamese Text

```
Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. 
Bà Lan, vợ ông Chúc, cũng làm việc tại đây.

Công ty TNHH MTV Việt Nam có trụ sở chính tại Hà Nội và chi nhánh tại Thành phố Hồ Chí Minh.

Chủ tịch nước Võ Văn Thưởng đã gặp gỡ các đại biểu trẻ tiêu biểu toàn quốc.
```

## Architecture

```
VnCoreNLP-Python-GUI/
├── vncorenlp_gui.py      # Main application file
└── README.md             # This file

VnCoreNLPProcessor Class:
- Interfaces with VnCoreNLP Java library via pyjnius
- Handles text annotation and result extraction
- Provides methods for specific NLP tasks

VnCoreNLPGUI Class:
- Manages tkinter UI components
- Handles user interactions
- Formats and displays results
- Implements async processing
```

## Troubleshooting

### "VnCoreNLP not initialized" Error

1. Ensure Java 11+ is installed
2. Verify `JAVA_HOME` is set correctly
3. Check that `VnCoreNLP-1.2.jar` exists in `__VnCoreNLP/`
4. Increase Java heap size if needed (modify `jnius_config.add_options`)

### "ImportError: No module named 'jnius'"

Install pyjnius:
```bash
pip install pyjnius
```

### GUI Doesn't Display

Ensure tkinter is installed:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS
brew install python-tk

# Windows
# tkinter comes with Python installer
```

### Memory Issues

For large texts, increase Java heap size in the code:
```python
jnius_config.add_options('-Xmx8g', '-Xms2g')  # Increase from default 4g/1g
```

## Performance Tips

- Process texts in chunks for very large documents
- Disable unused annotations for faster processing
- Use plain format for quick previews
- Close other applications when processing large files

## License

This Python GUI application is built on top of the VnCoreNLP library. Please refer to the original VnCoreNLP license for usage terms and conditions.

## Credits

- **VnCoreNLP Library**: Original Vietnamese NLP library
- **pyjnius**: Python-Java bridge library
- **tkinter**: Python's standard GUI toolkit

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

For issues related to:
- **VnCoreNLP functionality**: Refer to original VnCoreNLP documentation
- **Python GUI**: Create an issue in this repository
- **pyjnius**: Refer to pyjnius documentation

---

**Version**: 1.0.0  
**Last Updated**: 2024
