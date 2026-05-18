#!/usr/bin/env python3
"""
VnCoreNLP Flask Web API

A RESTful API service with an interactive web interface for Vietnamese NLP processing.

Features:
- Word Segmentation (wseg)
- POS Tagging (pos)
- Named Entity Recognition (ner)
- Dependency Parsing (parse)

API Endpoints:
- POST /api/v1/process - Process text with specified annotations
- POST /api/v1/entities - Extract named entities only
- GET /api/v1/health - Health check endpoint
- GET /api/v1/annotators - List available annotators
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from functools import wraps

# Configure JAVA_HOME and CLASSPATH
JAVA_HOME = os.environ.get('JAVA_HOME', '/usr/lib/jvm/java-17-openjdk-amd64')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VN_CORE_NLP_JAR = os.path.join(os.path.dirname(SCRIPT_DIR), '__VnCoreNLP', 'VnCoreNLP-1.2.jar')

os.environ['JAVA_HOME'] = JAVA_HOME

import jnius_config
jnius_config.add_options('-Xmx4g', '-Xms1g')
jnius_config.set_classpath(VN_CORE_NLP_JAR)

from jnius import autoclass

# Check if Flask is available, install if needed
try:
    from flask import Flask, request, jsonify, render_template_string, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
    from flask import Flask, request, jsonify, render_template_string, send_from_directory
    from flask_cors import CORS


class VnCoreNLPService:
    """Service class for VnCoreNLP operations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.annotator = None
        self._initialize()
        self._initialized = True
    
    def _initialize(self):
        """Initialize the VnCoreNLP annotator."""
        try:
            Properties = autoclass('java.util.Properties')
            Pipeline = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
            
            props = Properties()
            props.setProperty('annotators', 'wseg,pos,ner,parse')
            props.setProperty('ssplit.boundaryTokenRegex', r'[.।!?;:]+')
            props.setProperty('tokenize.language', 'vi')
            
            self.annotator = Pipeline(props)
            print("✓ VnCoreNLP service initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing VnCoreNLP: {e}")
            raise
    
    def process_text(self, text: str, annotators: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process Vietnamese text with specified annotations."""
        if not self.annotator:
            raise RuntimeError("VnCoreNLP not initialized")
        
        Annotation = autoclass('edu.stanford.nlp.pipeline.Annotation')
        CoreAnnotations = autoclass('edu.stanford.nlp.ling.CoreAnnotations')
        
        annotation = Annotation(text)
        self.annotator.annotate(annotation)
        
        result = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'sentences': []
        }
        
        sentences = annotation.get(CoreAnnotations.SentencesAnnotation).class_
        
        for i in range(sentences.size()):
            sentence_data = self._extract_sentence_data(sentences.get(i), CoreAnnotations)
            result['sentences'].append(sentence_data)
        
        result['entities'] = self._extract_entities(annotation, CoreAnnotations)
        
        return result
    
    def _extract_sentence_data(self, sentence, CoreAnnotations) -> Dict[str, Any]:
        """Extract data from a single sentence."""
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


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize service
nlp_service = None


def init_service():
    """Initialize NLP service."""
    global nlp_service
    try:
        nlp_service = VnCoreNLPService()
        return True
    except Exception as e:
        print(f"Failed to initialize service: {e}")
        return False


def require_service(f):
    """Decorator to check if service is initialized."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not nlp_service:
            return jsonify({'error': 'Service not initialized'}), 503
        return f(*args, **kwargs)
    return decorated_function


# HTML Template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VnCoreNLP Web API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { opacity: 0.9; }
        .card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .options {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }
        .option {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .option input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .output {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            max-height: 500px;
            overflow-y: auto;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.6;
        }
        .entity {
            display: inline-block;
            padding: 3px 8px;
            margin: 2px;
            border-radius: 4px;
            font-weight: bold;
        }
        .entity-PER { background: #ffcdd2; color: #c62828; }
        .entity-ORG { background: #bbdefb; color: #1565c0; }
        .entity-LOC { background: #c8e6c9; color: #2e7d32; }
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .actions button {
            padding: 8px 16px;
            font-size: 14px;
        }
        .api-info {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .api-info code {
            background: #34495e;
            padding: 2px 6px;
            border-radius: 3px;
        }
        @media (max-width: 768px) {
            header h1 { font-size: 1.8em; }
            .options { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🇻🇳 VnCoreNLP Web API</h1>
            <p>Vietnamese Natural Language Processing Service</p>
        </header>
        
        <div class="card">
            <h2>Input Text</h2>
            <textarea id="inputText" placeholder="Enter Vietnamese text here...&#10;&#10;Example: Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội."></textarea>
            
            <div class="options">
                <div class="option">
                    <input type="checkbox" id="wseg" checked>
                    <label for="wseg">Word Segmentation</label>
                </div>
                <div class="option">
                    <input type="checkbox" id="pos" checked>
                    <label for="pos">POS Tagging</label>
                </div>
                <div class="option">
                    <input type="checkbox" id="ner" checked>
                    <label for="ner">Named Entity Recognition</label>
                </div>
                <div class="option">
                    <input type="checkbox" id="parse" checked>
                    <label for="parse">Dependency Parsing</label>
                </div>
            </div>
            
            <button id="processBtn" onclick="processText()">⚡ Process Text</button>
        </div>
        
        <div class="card" id="resultCard" style="display: none;">
            <h2>Results</h2>
            <div class="output" id="output"></div>
            <div class="stats" id="stats"></div>
            <div class="actions">
                <button onclick="copyResults()">📋 Copy</button>
                <button onclick="downloadJSON()">💾 Download JSON</button>
            </div>
        </div>
        
        <div class="card api-info">
            <h2>API Endpoints</h2>
            <p style="margin: 10px 0;"><code>POST /api/v1/process</code> - Process text with annotations</p>
            <p style="margin: 10px 0;"><code>POST /api/v1/entities</code> - Extract named entities only</p>
            <p style="margin: 10px 0;"><code>GET /api/v1/health</code> - Health check</p>
            <p style="margin: 10px 0;"><code>GET /api/v1/annotators</code> - List available annotators</p>
        </div>
    </div>
    
    <script>
        let currentResult = null;
        
        async function processText() {
            const text = document.getElementById('inputText').value.trim();
            if (!text) {
                alert('Please enter some text!');
                return;
            }
            
            const btn = document.getElementById('processBtn');
            btn.disabled = true;
            btn.textContent = 'Processing...';
            
            const annotators = [];
            if (document.getElementById('wseg').checked) annotators.push('wseg');
            if (document.getElementById('pos').checked) annotators.push('pos');
            if (document.getElementById('ner').checked) annotators.push('ner');
            if (document.getElementById('parse').checked) annotators.push('parse');
            
            try {
                const response = await fetch('/api/v1/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, annotators })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentResult = data;
                    displayResults(data);
                } else {
                    showError(data.error || 'Processing failed');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.textContent = '⚡ Process Text';
            }
        }
        
        function displayResults(data) {
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            const resultCard = document.getElementById('resultCard');
            
            // Format output
            let html = '<div style="margin-bottom: 20px;">';
            html += '<strong>Input:</strong> ' + escapeHtml(data.text) + '</div>';
            
            data.sentences.forEach((sent, i) => {
                html += '<div style="margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px;">';
                html += '<strong>Sentence ' + (i + 1) + ':</strong> ';
                
                sent.tokens.forEach(token => {
                    let className = 'entity';
                    if (token.ner && token.ner !== 'O') {
                        className += ' entity-' + token.ner;
                        html += '<span class="' + className + '">' + escapeHtml(token.word) + 
                                ' [' + token.ner + ']</span> ';
                    } else {
                        html += '<span class="entity" style="background: #e3f2fd;">' + 
                                escapeHtml(token.word) + 
                                ' <small style="color: #666;">[' + token.pos + ']</small></span> ';
                    }
                });
                
                html += '</div>';
            });
            
            if (data.entities && data.entities.length > 0) {
                html += '<div style="margin-top: 20px; padding: 15px; background: #fff3e0; border-radius: 8px;">';
                html += '<strong>Named Entities:</strong><br>';
                data.entities.forEach(entity => {
                    html += '<span class="entity entity-' + entity.type + '">' + 
                            escapeHtml(entity.text) + ' [' + entity.type + ']</span> ';
                });
                html += '</div>';
            }
            
            output.innerHTML = html;
            
            // Stats
            const numSentences = data.sentences.length;
            const numTokens = data.sentences.reduce((sum, s) => sum + s.tokens.length, 0);
            const numEntities = data.entities ? data.entities.length : 0;
            
            stats.innerHTML = `
                <div class="stat">
                    <div class="stat-value">${numSentences}</div>
                    <div class="stat-label">Sentences</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${numTokens}</div>
                    <div class="stat-label">Words</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${numEntities}</div>
                    <div class="stat-label">Entities</div>
                </div>
            `;
            
            resultCard.style.display = 'block';
        }
        
        function showError(message) {
            const output = document.getElementById('output');
            output.innerHTML = '<div class="error">Error: ' + escapeHtml(message) + '</div>';
            document.getElementById('resultCard').style.display = 'block';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function copyResults() {
            if (!currentResult) return;
            navigator.clipboard.writeText(JSON.stringify(currentResult, null, 2));
            alert('Results copied to clipboard!');
        }
        
        function downloadJSON() {
            if (!currentResult) return;
            const blob = new Blob([JSON.stringify(currentResult, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'vncorenlp_result_' + new Date().toISOString().slice(0, 10) + '.json';
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // Load sample text
        document.getElementById('inputText').value = 'Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây.\\n\\nCông ty TNHH MTV Việt Nam có trụ sở chính tại Hà Nội và chi nhánh tại Thành phố Hồ Chí Minh.';
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/v1/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if nlp_service else 'unhealthy',
        'service': 'VnCoreNLP Web API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/v1/annotators')
@require_service
def list_annotators():
    """List available annotators."""
    return jsonify({
        'annotators': [
            {'id': 'wseg', 'name': 'Word Segmentation', 'description': 'Split Vietnamese text into words'},
            {'id': 'pos', 'name': 'POS Tagging', 'description': 'Assign part-of-speech tags'},
            {'id': 'ner', 'name': 'Named Entity Recognition', 'description': 'Identify named entities'},
            {'id': 'parse', 'name': 'Dependency Parsing', 'description': 'Parse grammatical structure'}
        ]
    })


@app.route('/api/v1/process', methods=['POST'])
@require_service
def process():
    """Process text with specified annotations."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data['text']
        annotators = data.get('annotators', ['wseg', 'pos', 'ner', 'parse'])
        
        result = nlp_service.process_text(text, annotators)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/entities', methods=['POST'])
@require_service
def extract_entities():
    """Extract named entities only."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data['text']
        result = nlp_service.process_text(text, ['wseg', 'pos', 'ner'])
        
        return jsonify({
            'text': text,
            'entities': result['entities'],
            'count': len(result['entities'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VnCoreNLP Flask Web API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VnCoreNLP Web API Server")
    print("=" * 60)
    
    # Initialize service
    print("Initializing VnCoreNLP service...")
    if not init_service():
        print("Failed to initialize service. Exiting.")
        sys.exit(1)
    
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("\nAvailable endpoints:")
    print("  GET  /                    - Web interface")
    print("  GET  /api/v1/health       - Health check")
    print("  GET  /api/v1/annotators   - List annotators")
    print("  POST /api/v1/process      - Process text")
    print("  POST /api/v1/entities     - Extract entities")
    print("=" * 60)
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
