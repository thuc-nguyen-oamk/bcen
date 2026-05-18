# VnCoreNLP Real-World Applications

This repository contains three real-world applications built on top of the VnCoreNLP Vietnamese NLP library, featuring:
- **Word Segmentation** (wseg)
- **POS Tagging** (pos)
- **Named Entity Recognition** (ner)
- **Dependency Parsing** (parse)

## Applications Overview

### 1. VnCoreNLP-GUI (Desktop Application)
A modern Swing-based desktop application with a graphical user interface for interactive text analysis.

**Features:**
- 🖥️ Modern, intuitive GUI with color-coded panels
- ⚡ Real-time text processing with progress indicator
- 📊 Statistics display (sentence count, word count)
- 📋 Copy/paste from clipboard support
- 💾 Save results in multiple formats (TXT, JSON, HTML)
- 📂 Load text files for batch analysis
- 🎨 Multiple output formats: Tabular, JSON, Plain Text, HTML
- ✅ Selectable annotation types

**Build & Run:**
```bash
cd VnCoreNLP-GUI
mvn clean package
java -jar target/VnCoreNLP-GUI-1.0.0.jar
```

### 2. VnCoreNLP-BatchProcessor (Command-Line Tool)
A high-performance command-line tool for processing large volumes of text files.

**Features:**
- 🚀 Parallel processing with configurable thread count
- 📁 Recursive directory scanning for text files
- 📄 Multiple output formats: Tabular, JSON, CoNLL
- 📊 Progress reporting with success/failure counts
- ⚙️ Flexible annotation configuration
- 🔧 Command-line argument parsing

**Usage:**
```bash
cd VnCoreNLP-BatchProcessor
mvn clean package

# Basic usage
java -jar target/VnCoreNLP-BatchProcessor-1.0.0.jar -i ./input -o ./output

# With custom options
java -jar target/VnCoreNLP-BatchProcessor-1.0.0.jar \
    --input ./documents \
    --output ./results \
    --format json \
    --threads 8

# Disable specific annotations
java -jar target/VnCoreNLP-BatchProcessor-1.0.0.jar \
    -i ./input -o ./output \
    --disable-parse --disable-ner
```

**Command-Line Options:**
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

### 3. VnCoreNLP-WebAPI (REST API + Web UI)
A Spring Boot REST API service with an interactive web interface.

**Features:**
- 🌐 RESTful API endpoints for all NLP functions
- 🖥️ Beautiful, responsive web interface
- 📱 Mobile-friendly design
- 🔗 CORS-enabled for cross-origin requests
- 📊 Interactive entity visualization
- 💾 Download results as JSON or TXT
- 📋 One-click copy to clipboard
- 🏥 Health check endpoint

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/process` | POST | Process text with specified annotations (JSON response) |
| `/api/v1/process/tabular` | POST | Process text and return tabular format |
| `/api/v1/entities` | POST | Extract named entities only |
| `/api/v1/health` | GET | Health check endpoint |
| `/api/v1/annotators` | GET | List available annotators |

**Example API Request:**
```bash
curl -X POST http://localhost:8080/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.",
    "annotators": ["wseg", "pos", "ner", "parse"]
  }'
```

**Build & Run:**
```bash
cd VnCoreNLP-WebAPI
mvn clean package
java -jar target/VnCoreNLP-WebAPI-1.0.0.jar
```

**Access the Web Interface:**
Open your browser and navigate to: `http://localhost:8080`

**API Documentation:**
Swagger UI available at: `http://localhost:8080/swagger-ui.html`

## Project Structure

```
/workspace/
├── __VnCoreNLP/                    # Original VnCoreNLP library
│   ├── VnCoreNLP-1.2.jar
│   ├── models/
│   └── src/
│
├── VnCoreNLP-GUI/                  # Desktop GUI Application
│   ├── pom.xml
│   └── src/main/java/gui/
│       └── VnCoreNLPApp.java
│
├── VnCoreNLP-BatchProcessor/       # Command-Line Batch Tool
│   ├── pom.xml
│   └── src/main/java/batch/
│       └── BatchProcessor.java
│
└── VnCoreNLP-WebAPI/               # REST API + Web Interface
    ├── pom.xml
    └── src/
        ├── main/java/webapi/
        │   ├── VnCoreNLPWebAPI.java
        │   ├── controller/
        │   │   └── NLPController.java
        │   └── service/
        │       └── NLPService.java
        └── main/resources/static/
            └── index.html
```

## Requirements

- Java 11 or higher
- Maven 3.6+
- The VnCoreNLP-1.2.jar must be present in `__VnCoreNLP/` directory

## Sample Vietnamese Text for Testing

```
Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. 
Bà Lan, vợ ông Chúc, cũng làm việc tại đây.

Công ty TNHH MTV Việt Nam có trụ sở chính tại Hà Nội và chi nhánh tại Thành phố Hồ Chí Minh.
```

## License

These applications are built on top of the VnCoreNLP library. Please refer to the original VnCoreNLP license for usage terms.
