import os
import json
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from markupsafe import escape

# Try to import WHOIS for the domain age enhancement
try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

app = Flask(__name__)

# --- 1. LOAD THE AI BRAIN ---
try:
    # Ensure you ran train_ai.py first to generate these!
    model = joblib.load('phishing.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ AI Engine Online: Hybrid Model Loaded")
except Exception as e:
    print(f"❌ ERROR: AI Models not found. Run train_ai.py first. Detail: {e}")

LOG_FILE = 'scan_history.json'

def get_domain_age(url):
    """Fetches how many days old a domain is"""
    if not WHOIS_AVAILABLE:
        return None
    try:
        domain_name = url.split("//")[-1].split("/")[0]
        w = whois.whois(domain_name)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            return (datetime.now() - creation_date).days
        return None
    except:
        return None

def save_to_logs(url, status, score):
    """Saves scan to history, preventing duplicate URLs"""
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except: 
            logs = []
    
    # Check if this URL has been scanned before
    existing_entry = next((item for item in logs if item['url'] == str(escape(url))), None)

    if existing_entry:
        # UPDATE the existing record instead of adding a new one
        existing_entry['status'] = status
        existing_entry['score'] = score
        existing_entry['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔄 Updated existing log for: {url}")
    else:
        # ADD a brand new record
        logs.append({
            'url': str(escape(url)),
            'status': status,
            'score': score,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"📝 Created new log for: {url}")

    # Save the updated list
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/logs')
def get_logs():
    """Groups logs into Good and Bad for the UI columns"""
    if not os.path.exists(LOG_FILE):
        return jsonify({'good': [], 'bad': []})
    with open(LOG_FILE, 'r') as f:
        data = json.load(f)
    
    # Newest first
    data.reverse()
    return jsonify({
        'good': [i for i in data if i['status'] == 'safe'],
        'bad': [i for i in data if i['status'] != 'safe']
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        url = data.get('url', '')
        if not url: return jsonify({'status': 'error', 'message': 'Empty URL'})

        # --- 1. GET DYNAMIC AI PROBABILITY ---
        vec = vectorizer.transform([url])
        
        # This gets the probability for [Safe, Phishing]
        # Example output: [[0.05, 0.95]] means 95% Phishing
        probabilities = model.predict_proba(vec)[0] 
        
        # We take the probability of it being 'bad'
        # Note: Depending on your training, 'bad' might be index 0 or 1.
        # We'll use the model's classes to be sure.
        bad_index = list(model.classes_).index('bad')
        ai_confidence = float(probabilities[bad_index]) * 100

        # --- 2. WHOIS Enhancement ---
        age = get_domain_age(url)
        
        # --- 3. DYNAMIC SCORING LOGIC ---
        # Start with the AI's raw confidence
        risk_score = ai_confidence
        reasons = []

        if ai_confidence > 50:
            reasons.append(f"AI Model is {ai_confidence:.1f}% confident this matches phishing patterns")
        else:
            reasons.append(f"AI Analysis indicates a {100 - ai_confidence:.1f}% safety confidence")

        # Adjust score based on age (The "Hybrid" factor)
        if age is not None:
            if age < 30:
                risk_score += 15 # Add risk for very new domains
                reasons.append(f"Security Alert: Domain is brand new ({age} days old)")
            elif age > 1000:
                risk_score -= 10 # Reduce risk for very old domains
                reasons.append("Trust Factor: Established domain (over 3 years old)")
        
        # Keep score between 0 and 100
        final_score = max(0, min(100, round(risk_score, 1)))
        
        status = "danger" if final_score >= 50 else "safe"
        
        save_to_logs(url, status, final_score)

        return jsonify({
            'status': status,
            'score': final_score,
            'reasons': reasons,
            'age': f"{age} days" if age else "Unknown"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)