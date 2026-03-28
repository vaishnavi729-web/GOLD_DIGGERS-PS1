import os
import re
import json
import sqlite3
import requests
import base64
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
import easyocr
from datetime import datetime

app = Flask(__name__, static_folder='frontend', template_folder='frontend')
CORS(app)

# --- Configuration & Model Loading ---
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Global models
print("Loading ClimateBERT...")
try:
    climate_pipe = pipeline("text-classification", model="climatebert/distilroberta-base-climate-fever", device=-1)
except Exception as e:
    print(f"Error loading ClimateBERT: {e}")
    climate_pipe = None

print("Loading Sentence Transformer...")
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading Sentence Transformer: {e}")
    similarity_model = None

print("Initializing EasyOCR...")
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Error loading EasyOCR: {e}")
    reader = None

# --- Database & Data Helpers ---
def init_db():
    conn = sqlite3.connect('data/reports.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  brand_name TEXT,
                  reason TEXT,
                  evidence_url TEXT,
                  created_at TEXT)''')
    conn.commit()
    conn.close()

def load_certified_brands():
    try:
        df = pd.read_csv('data/certified_brands.csv')
        return df
    except Exception as e:
        print(f"Error loading brands: {e}")
        return pd.DataFrame()

# Initialize on startup
init_db()
BRANDS_DF = load_certified_brands()

# --- Core Logic Functions ---
def detect_buzzwords(text):
    buzzwords = {
        'natural': '⚠️ Vague term - no legal definition',
        'green': '⚠️ Marketing buzzword - requires evidence',
        'eco-friendly': '⚠️ Unverifiable without certification',
        'sustainable': '⚠️ Needs specific evidence',
        'earth-friendly': '⚠️ Marketing fluff',
        'non-toxic': '⚠️ Check for specific claims',
        'biodegradable': '⚠️ Requires conditions',
        'organic': '✅ Good if certified'
    }
    found = []
    text_lower = text.lower()
    for word, explanation in buzzwords.items():
        if word in text_lower:
            found.append({'word': word, 'explanation': explanation})
    return found

def classify_with_climatebert(text):
    if not climate_pipe:
        return {'evidence_confidence': 0.5, 'classification': 'Unknown (Model Load Error)'}
    
    # Clip text for model length
    input_text = text[:512]
    result = climate_pipe(input_text)[0]
    # Climate Fever labels: SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    # Simplification for score: 
    score = result['score']
    label = result['label']
    
    mapping = {
        'SUPPORTS': ('Evidence-Based', score),
        'REFUTES': ('Marketing Fluff', 1 - score),
        'NOT_ENOUGH_INFO': ('Vague Claim', 0.5)
    }
    
    status, confidence = mapping.get(label, ('Unknown', 0.5))
    return {'evidence_confidence': confidence, 'classification': status}

def check_contradictions(text):
    # Simple semantic contradiction check
    contradictions = [
        ("natural", "synthetic"),
        ("organic", "polyester"),
        ("eco-friendly", "plastic"),
        ("biodegradable", "polyester"),
        ("natural", "polymers")
    ]
    text_lower = text.lower()
    for word1, word2 in contradictions:
        if word1 in text_lower and word2 in text_lower:
            return {
                'has_contradiction': True, 
                'explanation': f"Found contradiction: Product claims to be '{word1}' but contains '{word2}'."
            }
    return {'has_contradiction': False, 'explanation': ''}

def detect_certifications(text):
    patterns = {
        'GOTS': r'GOTS|Global Organic Textile Standard',
        'B-Corp': r'B Corp|B Corporation|BCorp',
        'USDA Organic': r'USDA Organic',
        'Fair Trade': r'Fair Trade|Fairtrade',
        'Rainforest Alliance': r'Rainforest Alliance'
    }
    found = []
    for cert, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(cert)
    return found

def calculate_score(evidence_score, buzzword_count, has_cert, has_contradiction):
    # Formula: 40% evidence, 20% buzzwords, 25% cert, 15% contradiction
    e_contrib = (evidence_score * 100) * 0.4
    
    b_penalty = min(buzzword_count * 5, 20)
    b_contrib = (20 - b_penalty)
    
    cert_contrib = 25 if has_cert else 0
    
    contra_contrib = 0 if has_contradiction else 15
    
    total = e_contrib + b_contrib + cert_contrib + contra_contrib
    return round(min(max(total, 0), 100))

def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try metadata meta tags first
        desc = soup.find('meta', attrs={'name': 'description'}) or \
               soup.find('meta', attrs={'property': 'og:description'})
        if desc:
            return desc.get('content', '')
            
        # fallback to common text
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs[:5]])
        return text[:2000]
    except Exception as e:
        return f"Scraping Error: {str(e)}"

def extract_text_from_image(image_base64):
    if not reader:
        return "EasyOCR Error: Model not loaded"
    
    try:
        # base64 to image
        img_data = base64.b64decode(image_base64.split(',')[-1])
        img = Image.open(io.BytesIO(img_data))
        img_np = np.array(img)
        
        # run ocr
        results = reader.readtext(img_np)
        extracted = ' '.join([res[1] for res in results])
        return extracted
    except Exception as e:
        return f"OCR Error: {str(e)}"

# --- API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    input_type = data.get('type')
    content = data.get('content')
    
    processed_text = ""
    if input_type == 'text':
        processed_text = content
    elif input_type == 'url':
        processed_text = scrape_url(content)
    elif input_type == 'image':
        processed_text = extract_text_from_image(content)
    
    if not processed_text:
        return jsonify({'error': 'No text found or processing failed'}), 400
        
    buzzwords = detect_buzzwords(processed_text)
    climate_res = classify_with_climatebert(processed_text)
    contra_res = check_contradictions(processed_text)
    certs = detect_certifications(processed_text)
    
    score = calculate_score(
        climate_res['evidence_confidence'], 
        len(buzzwords), 
        len(certs) > 0, 
        contra_res['has_contradiction']
    )
    
    return jsonify({
        'text': processed_text,
        'buzzwords': buzzwords,
        'climate_analysis': climate_res,
        'contradiction': contra_res,
        'certifications': certs,
        'score': score,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/alternatives', methods=['POST'])
def get_alternatives():
    if BRANDS_DF.empty or not similarity_model:
        return jsonify([])
        
    data = request.json
    product_text = data.get('text', '')
    
    # Simple semantic search
    product_emb = similarity_model.encode(product_text, convert_to_tensor=True)
    brand_texts = (BRANDS_DF['brand_name'] + " " + BRANDS_DF['description']).tolist()
    brand_embs = similarity_model.encode(brand_texts, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(product_emb, brand_embs)[0]
    top_results = torch.topk(cosine_scores, k=min(3, len(BRANDS_DF)))
    
    indices = top_results.indices.tolist()
    results = BRANDS_DF.iloc[indices].to_dict('records')
    return jsonify(results)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').lower()
    context = data.get('context', {})
    
    score = context.get('score', 0)
    buzzwords = context.get('buzzwords', [])
    certs = context.get('certifications', [])
    
    if 'score' in message or 'rating' in message:
        response = f"The product's transparency score is {score}/100. "
        if score > 70:
            response += "It looks very reliable!"
        elif score > 40:
            response += "It has some good points, but watch out for vague claims."
        else:
            response += "This is likely greenwashing; proceed with caution."
    elif 'buzzword' in message or 'claim' in message:
        if buzzwords:
            words = ", ".join([b['word'] for b in buzzwords])
            response = f"I detected several buzzwords: {words}. Brands use these terms to sound eco-friendly without always being specific."
        else:
            response = "I didn't find many suspicious buzzwords. This might be a more transparent claim."
    elif 'certif' in message:
        if certs:
            response = f"Yes, I detected {', '.join(certs)}. These are great indicators of genuine sustainability."
        else:
            response = "I couldn't verify any major certifications. Look for symbols like B-Corp, GOTS, or USDA Organic on the packaging."
    elif 'alternative' in message or 'better' in message:
        response = "You can scroll down to the 'Certified Alternatives' section to find verified sustainable brands in similar categories!"
    else:
        response = "I can help explain your score, discuss the buzzwords found, or verify certifications. What would you like to know?"
        
    return jsonify({'response': response})

@app.route('/api/report', methods=['POST'])
def report():
    data = request.json
    conn = sqlite3.connect('data/reports.db')
    c = conn.cursor()
    c.execute("INSERT INTO reports (brand_name, reason, evidence_url, created_at) VALUES (?, ?, ?, ?)",
              (data.get('brand_name'), data.get('reason'), data.get('evidence_url'), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/api/reports', methods=['GET'])
def get_reports():
    conn = sqlite3.connect('data/reports.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM reports ORDER BY created_at DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
