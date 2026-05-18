# VnCoreNLP Python Applications

A collection of real-world Python applications built on top of the VnCoreNLP Vietnamese NLP library.

## Overview

This repository contains three Python applications that leverage VnCoreNLP for Vietnamese Natural Language Processing:

| Application | Type | Description |
|-------------|------|-------------|
| **VnCoreNLP-Python-GUI** | Desktop GUI | Modern tkinter-based desktop application |
| **VnCoreNLP-Python-Batch** | CLI Tool | Command-line batch processor for large files |
| **VnCoreNLP-Python-WebAPI** | Web Service | RESTful API with interactive web interface |

## Features

All applications support the following NLP features:

✅ **Word Segmentation** (wseg) - Split Vietnamese text into words  
✅ **POS Tagging** (pos) - Assign part-of-speech tags to words  
✅ **Named Entity Recognition** (ner) - Identify persons, organizations, locations  
✅ **Dependency Parsing** (parse) - Analyze grammatical structure  

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install pyjnius flask flask-cors

# Verify Java installation
java -version  # Should be Java 11+
```

### 1. Desktop GUI Application

```bash
cd VnCoreNLP-Python-GUI
python vncorenlp_gui.py
```

**Best for**: Interactive text analysis, quick testing, educational purposes

### 2. Batch Processor (CLI)

```bash
cd VnCoreNLP-Python-Batch
python batch_processor.py -i ./input -o ./output --format json --threads 8
```

**Best for**: Processing large volumes of files, automation, data pipelines

### 3. Web API Service

```bash
cd VnCoreNLP-Python-WebAPI
python app.py --host 0.0.0.0 --port 8080
```

Then open http://localhost:8080 in your browser.

**Best for**: Web integration, microservices, remote access, API development

## Project Structure

```
/workspace/
├── __VnCoreNLP/                    # Original VnCoreNLP library (Java)
│   ├── VnCoreNLP-1.2.jar
│   └── models/
│
├── VnCoreNLP-Python-GUI/           # Desktop GUI Application
│   ├── vncorenlp_gui.py           # Main application
│   └── README.md
│
├── VnCoreNLP-Python-Batch/         # Command-Line Batch Tool
│   ├── batch_processor.py         # Main script
│   └── README.md
│
└── VnCoreNLP-Python-WebAPI/        # REST API + Web Interface
    ├── app.py                     # Flask application
    └── README.md
```

## Sample Vietnamese Text

Use this sample text for testing:

```
Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. 
Bà Lan, vợ ông Chúc, cũng làm việc tại đây.

Công ty TNHH MTV Việt Nam có trụ sở chính tại Hà Nội và 
chi nhánh tại Thành phố Hồ Chí Minh.

Chủ tịch nước Võ Văn Thưởng đã gặp gỡ các đại biểu trẻ tiêu biểu toàn quốc.
```

## Requirements Summary

- **Python**: 3.8 or higher
- **Java**: 11 or higher
- **Dependencies**:
  - pyjnius (all apps)
  - flask, flask-cors (Web API)
  - tkinter (GUI, usually included)

## License

These applications are built on top of the VnCoreNLP library. Please refer to the original VnCoreNLP license for usage terms and conditions.

---

**Version**: 1.0.0  
**Last Updated**: 2024
