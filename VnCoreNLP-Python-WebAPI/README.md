# VnCoreNLP Python Web API

A RESTful API service with an interactive web interface for Vietnamese NLP processing, built with Flask.

## Features

✅ **Word Segmentation** (wseg)  
✅ **POS Tagging** (pos)  
✅ **Named Entity Recognition** (ner)  
✅ **Dependency Parsing** (parse)  

### Key Features

- 🌐 RESTful API endpoints for all NLP functions
- 🖥️ Beautiful, responsive web interface
- 📱 Mobile-friendly design
- 🔗 CORS-enabled for cross-origin requests
- 📊 Interactive entity visualization
- 💾 Download results as JSON
- 📋 One-click copy to clipboard
- 🏥 Health check endpoint

## Requirements

- Python 3.8 or higher
- Java 11 or higher
- pyjnius
- Flask
- flask-cors

## Installation

```bash
pip install pyjnius flask flask-cors
```

## Usage

### Start the Server

```bash
cd VnCoreNLP-Python-WebAPI
python app.py
```

### Custom Host and Port

```bash
python app.py --host 0.0.0.0 --port 8080
```

### Debug Mode

```bash
python app.py --debug
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/v1/process` | POST | Process text with specified annotations |
| `/api/v1/entities` | POST | Extract named entities only |
| `/api/v1/health` | GET | Health check endpoint |
| `/api/v1/annotators` | GET | List available annotators |

## Example API Requests

### Process Text

```bash
curl -X POST http://localhost:8080/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.",
    "annotators": ["wseg", "pos", "ner", "parse"]
  }'
```

### Extract Entities Only

```bash
curl -X POST http://localhost:8080/api/v1/entities \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Công ty TNHH MTV Việt Nam có trụ sở tại Hà Nội."
  }'
```

### Health Check

```bash
curl http://localhost:8080/api/v1/health
```

### List Annotators

```bash
curl http://localhost:8080/api/v1/annotators
```

## Example Response

### POST /api/v1/process

```json
{
  "text": "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.",
  "timestamp": "2024-01-15T10:30:45.123456",
  "sentences": [
    {
      "text": "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.",
      "tokens": [
        {
          "index": 1,
          "word": "Ông",
          "lemma": "ông",
          "pos": "N",
          "ner": "O"
        },
        {
          "index": 2,
          "word": "Nguyễn",
          "lemma": "nguyễn",
          "pos": "N",
          "ner": "P_PER"
        }
      ],
      "dependencies": [
        {
          "governor": 2,
          "dependent": 1,
          "relation": "nsubj"
        }
      ]
    }
  ],
  "entities": [
    {
      "text": "Nguyễn Khắc Chúc",
      "type": "P_PER"
    },
    {
      "text": "Đại học Quốc gia Hà Nội",
      "type": "ORG"
    }
  ]
}
```

## Web Interface

Access the web interface by opening your browser to:

```
http://localhost:8080
```

### Features

1. **Text Input**: Enter or paste Vietnamese text
2. **Annotation Selection**: Choose which annotations to apply
3. **Visual Results**: Color-coded named entities and POS tags
4. **Statistics**: View sentence, word, and entity counts
5. **Export Options**: Copy to clipboard or download as JSON

### Sample Text

The web interface includes a sample text button that loads:

```
Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. 
Bà Lan, vợ ông Chúc, cũng làm việc tại đây.

Công ty TNHH MTV Việt Nam có trụ sở chính tại Hà Nội và 
chi nhánh tại Thành phố Hồ Chí Minh.
```

## Integration Examples

### Python Client

```python
import requests

def process_vietnamese_text(text):
    response = requests.post(
        'http://localhost:8080/api/v1/process',
        json={
            'text': text,
            'annotators': ['wseg', 'pos', 'ner']
        }
    )
    return response.json()

result = process_vietnamese_text("Xin chào Việt Nam!")
print(result)
```

### JavaScript Client

```javascript
async function processText(text) {
    const response = await fetch('http://localhost:8080/api/v1/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            annotators: ['wseg', 'pos', 'ner', 'parse']
        })
    });
    return await response.json();
}

processText("Xin chào Việt Nam!").then(console.log);
```

### cURL Examples

```bash
# Full processing
curl -X POST http://localhost:8080/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chào Việt Nam!", "annotators": ["wseg", "pos", "ner", "parse"]}'

# Entity extraction only
curl -X POST http://localhost:8080/api/v1/entities \
  -H "Content-Type: application/json" \
  -d '{"text": "Chủ tịch Võ Văn Thưởng gặp gỡ đại biểu tại Hà Nội"}'
```

## Configuration

### Environment Variables

- `JAVA_HOME`: Path to Java installation (auto-detected if not set)
- `VN_CORE_NLP_JAR`: Path to VnCoreNLP JAR file (default: `../__VnCoreNLP/VnCoreNLP-1.2.jar`)

### Memory Settings

For large texts, modify memory settings in `app.py`:

```python
jnius_config.add_options('-Xmx8g', '-Xms2g')  # Increase from default 4g/1g
```

## Troubleshooting

### Service Not Initialized

If you see "Service not initialized" error:
1. Ensure Java 11+ is installed
2. Verify `JAVA_HOME` is set correctly
3. Check that `VnCoreNLP-1.2.jar` exists

### CORS Issues

CORS is enabled by default. If you encounter issues, ensure:
- `flask-cors` is installed
- The server is running with CORS enabled

### Memory Errors

For large documents:
1. Increase Java heap size in the code
2. Process text in smaller chunks
3. Use batch processor for large files

## Performance Tips

1. **Use specific annotators**: Only request what you need
   ```json
   {"annotators": ["wseg", "ner"]}  // Skip POS and parse if not needed
   ```

2. **Batch requests**: For multiple texts, consider using the batch processor

3. **Connection pooling**: For production use, use a WSGI server like Gunicorn
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8080 app:app
   ```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y openjdk-17-jre-headless

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080"]
```

## License

This application is built on top of the VnCoreNLP library. Please refer to the original VnCoreNLP license for usage terms.

## Credits

- **VnCoreNLP Library**: Original Vietnamese NLP library
- **Flask**: Web framework
- **pyjnius**: Python-Java bridge

---

**Version**: 1.0.0  
**Last Updated**: 2024
