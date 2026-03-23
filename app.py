import os
import re
import json
import joblib
from datetime import datetime
from urllib.parse import urlparse
import pandas as pd

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import hstack, csr_matrix
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from flask import Flask, render_template, request, jsonify
from markupsafe import escape

nltk.download('stopwords', quiet=True)

# WHOIS (optional)
try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

# FLASK APP SETUP 
app = Flask(__name__)

# Loading the trained pipeline components from the pickle file
try:
    bundle      = joblib.load('phishing_pipeline_v2.pkl')
    model       = bundle['model']
    tfidf       = bundle['tfidf']
    scaler      = bundle['scaler']
    FEAT_COLS   = bundle['feature_columns']
    MODEL_NAME  = bundle['model_name']
    print(f"✅Pipeline loaded Model: {MODEL_NAME}")
except Exception as e:
    print(f"❌ ERROR: Could not load pipeline: {e}")
    bundle = None

# PREPROCESSING CONSTANTS 
KNOWN_BRANDS = [
    'paypal', 'google', 'facebook', 'apple', 'microsoft', 'amazon',
    'netflix', 'instagram', 'twitter', 'linkedin', 'ebay', 'dropbox',
    'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'dhl', 'fedex',
    'whatsapp', 'telegram', 'yahoo', 'outlook', 'office365', 'adobe'
]

SUSPICIOUS_TLDS = [
    'xyz', 'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'click', 'link',
    'online', 'site', 'info', 'biz', 'pw', 'cc', 'ws', 'su'
]

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'verify', 'secure', 'account', 'update',
    'banking', 'confirm', 'password', 'credential', 'suspend',
    'validate', 'alert', 'free', 'prize'
]

tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer   = SnowballStemmer('english')

LOG_FILE     = 'scan_history.json'
MAX_LOG_SIZE = 500   # This keep only the latest 500 entries

# PREPROCESSING FUNCTIONS 
def preprocess_url_text(url):
    """Tokenize and stem URL — must match exactly what was done during training."""
    tokens  = tokenizer.tokenize(str(url))
    stemmed = [stemmer.stem(t) for t in tokens]
    return ' '.join(stemmed)


def extract_features(url):
    """
    Extract the same 15 numerical features used during training.
    Column order must match FEAT_COLS exactly.
    """
    try:
        import tldextract
        ext       = tldextract.extract(url)
        domain    = ext.domain
        subdomain = ext.subdomain
        tld       = ext.suffix
    except Exception:
        domain = subdomain = tld = ''

    ip_pattern = r'(http[s]?://)?' r'(\d{1,3}\.){3}\d{1,3}'

    try:
        parsed     = urlparse(url if url.startswith('http') else 'http://' + url)
        path_depth = len([p for p in parsed.path.split('/') if p])
    except Exception:
        path_depth = 0

    brand_in_subdomain = any(b in subdomain for b in KNOWN_BRANDS)
    brand_is_domain    = any(b == domain    for b in KNOWN_BRANDS)

    return {
        'url_length'              : len(url),
        'dot_count'               : url.count('.'),
        'hyphen_count'            : url.count('-'),
        'slash_count'             : url.count('/'),
        'at_symbol'               : int('@' in url),
        'double_slash'            : int('//' in url[7:]),
        'https'                   : int(url.startswith('https')),
        'has_ip'                  : int(bool(re.match(ip_pattern, url))),
        'special_char_count'      : len(re.findall(r'[^a-zA-Z0-9/.:_-]', url)),
        'subdomain_depth'         : len(subdomain.split('.')) if subdomain else 0,
        'domain_length'           : len(domain),
        'suspicious_tld'          : int(tld in SUSPICIOUS_TLDS),
        'brand_spoofing'          : int(brand_in_subdomain and not brand_is_domain),
        'suspicious_keyword_count': sum(1 for w in SUSPICIOUS_KEYWORDS if w in url),
        'path_depth'              : path_depth,
    }


def build_reasons(url, feats, ai_confidence, age):
    """
    Building a human-readable list of reasons explaining why the URL
    received the score it did — shown to the user in the result card.
    """
    reasons = [f"AI Model Confidence: {ai_confidence:.1f}%"]

    if feats['has_ip']:
        reasons.append("⚠️ Raw IP address used as domain — no legitimate site does this")
    if feats['brand_spoofing']:
        reasons.append("⚠️ Known brand name found in subdomain but not as the real domain (spoofing)")
    if feats['suspicious_tld']:
        reasons.append("⚠️ Suspicious TLD detected (.tk, .xyz, .ml etc.) — commonly used in phishing")
    if feats['at_symbol']:
        reasons.append("⚠️ '@' symbol in URL — forces browser to ignore everything before it")
    if feats['double_slash']:
        reasons.append("⚠️ Double slash '//' detected after protocol — possible redirect trick")
    if not feats['https']:
        reasons.append("⚠️ No HTTPS — connection is unencrypted")
    if feats['suspicious_keyword_count'] >= 3:
        reasons.append(f"⚠️ {feats['suspicious_keyword_count']} suspicious keywords found (verify, login, secure…)")
    if feats['dot_count'] >= 4:
        reasons.append(f"⚠️ High dot count ({feats['dot_count']}) — deep subdomain nesting detected")
    if feats['url_length'] > 100:
        reasons.append(f"⚠️ URL is unusually long ({feats['url_length']} characters)")
    if age is not None and age < 30:
        reasons.append(f"⚠️ Domain is only {age} days old — very recently registered")
    elif age is not None and age > 1000:
        reasons.append(f" Domain is {age} days old — established and trusted")

    return reasons


def predict_url(url):
    
    if bundle is None:
        raise RuntimeError("Pipeline not loaded")

    # Step 1 & 2 — text features
    text      = preprocess_url_text(url)
    tfidf_vec = tfidf.transform([text])

    # Step 3 & 4 — numerical features
    feats      = extract_features(url)
    num_array  = pd.DataFrame([[feats[col] for col in FEAT_COLS]], columns=FEAT_COLS)
    num_scaled = scaler.transform(num_array)
    num_sparse = csr_matrix(num_scaled)

    # Step 5 — combine into hybrid matrix
    X = hstack([tfidf_vec, num_sparse])

    # Step 6 — predict
    proba    = model.predict_proba(X)[0]
    bad_idx  = list(model.classes_).index('bad')
    confidence = float(proba[bad_idx]) * 100

    return confidence, feats

# DOMAIN AGE (WHOIS) 
def get_domain_age(url):
    if not WHOIS_AVAILABLE:
        return None
    try:
        domain = url.replace('https://', '').replace('http://', '') \
                    .replace('www.', '').split('/')[0]
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            return (datetime.now() - creation_date).days
        return None
    except Exception:
        return None

#  URL VALIDATION 
def is_valid_url(url):
    """Basic sanity check before running the model."""
    if not url or len(url) < 4:
        return False
    # Must contain at least one dot
    if '.' not in url:
        return False
    # Block obviously local/private addresses
    private = ['localhost', '127.0.0.1', '0.0.0.0']
    if any(p in url for p in private):
        return False
    return True

# LOGGING 
def save_to_logs(url, status, score):
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except Exception:
            logs = []

    safe_url = str(escape(url))

    # Update existing entry if URL was scanned before
    existing = next((item for item in logs if item['url'] == safe_url), None)
    if existing:
        existing['status'] = status
        existing['score']  = score
        existing['time']   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        logs.append({
            'url'   : safe_url,
            'status': status,
            'score' : score,
            'time'  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Trim to latest MAX_LOG_SIZE entries to prevent unbounded growth
    if len(logs) > MAX_LOG_SIZE:
        logs = logs[-MAX_LOG_SIZE:]

    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)

# ROUTES
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/logs')
def get_logs():
    if not os.path.exists(LOG_FILE):
        return jsonify({'good': [], 'bad': []})
    with open(LOG_FILE, 'r') as f:
        data = json.load(f)
    data.reverse()
    return jsonify({
        'good': [i for i in data if i['status'] == 'safe'],
        'bad' : [i for i in data if i['status'] != 'safe']
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data received'})

        url = data.get('url', '').strip().lower()

        #  Validate URL 
        if not url:
            return jsonify({'status': 'error', 'message': 'Please enter a URL'})
        if not is_valid_url(url):
            return jsonify({'status': 'error', 'message': 'Invalid URL format'})

        # Check pipeline loaded 
        if bundle is None:
            return jsonify({'status': 'error', 'message': 'AI model not loaded. Please check server logs.'})

        #  Run full prediction pipeline 
        ai_confidence, feats = predict_url(url)

        #  Domain age check 
        age        = get_domain_age(url)
        risk_score = ai_confidence

        # Score based on domain age
        if age is not None:
            if age < 30:
                risk_score += 15    # very new domain — extra suspicious
            elif age > 1000:
                risk_score -= 10   # well-established domain — slight boost

        # Hard boosts for the most reliable phishing signals
        if feats['has_ip']:
            risk_score += 20
        if feats['brand_spoofing']:
            risk_score += 15
        if feats['suspicious_tld']:
            risk_score += 10

        final_score = max(0, min(100, round(risk_score, 1)))
        status      = 'danger' if final_score >= 50 else 'safe'
        reasons     = build_reasons(url, feats, ai_confidence, age)

        save_to_logs(url, status, final_score)

        return jsonify({
            'status' : status,
            'score'  : final_score,
            'reasons': reasons,
            'age'    : f"{age} days" if age is not None else "Unknown",
            'model'  : MODEL_NAME,
            'features': {
                'has_ip'        : bool(feats['has_ip']),
                'brand_spoofing': bool(feats['brand_spoofing']),
                'suspicious_tld': bool(feats['suspicious_tld']),
                'https'         : bool(feats['https']),
                'dot_count'     : feats['dot_count'],
                'url_length'    : feats['url_length'],
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)