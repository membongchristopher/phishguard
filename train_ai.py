import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("⏳ Loading dataset... (This might take a minute as the file is large)")
# Load the dataset you got from GitHub
df = pd.read_csv('phishing_site_urls.csv')

print("✂️ Filtering data...")
# We take a subset (e.g., 100,000 rows) to make it fast, or remove this to use all 500k
df = df.sample(100000, random_state=42) 

print("🧪 Vectorizing text data...")
# Using TfidfVectorizer (This is what the GitHub project used)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['URL'])
y = df['Label'] # 'bad' or 'good'

print("🤖 Training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("💾 Saving your new compatible models...")
joblib.dump(model, 'phishing.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ SUCCESS! Your 'phishing.pkl' and 'vectorizer.pkl' are now ready and compatible.")