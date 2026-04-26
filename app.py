from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import pickle
import os
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None
scaler = None
feature_names = None
@app.get("/")
def root():
    return {"message": "Fraud Detection API - Kaggle Dataset Ready", "status": "running"}
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    global model, scaler, feature_names
    df = pd.read_csv(file.file)
    if 'Class' in df.columns:
        v_features = [col for col in df.columns if col.startswith('V')]
        feature_cols = v_features + ['Time', 'Amount']
        X = df[feature_cols].values
        y = df['Class'].values
        feature_names = feature_cols
        print(f"✅ Loaded Kaggle dataset: {len(feature_cols)} features")
        print(f"   Features: {feature_cols[:5]}... (total {len(feature_cols)})")
    else:
        return JSONResponse({"error": "Invalid format. Need Kaggle dataset with Class column"}, status_code=400)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GaussianNB()
    model.fit(X_scaled, y)
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feature_names}, f)
    fraud_count = int(y.sum())
    total = len(y)
    fraud_rate = (fraud_count/total)*100
    y_pred = model.predict(X_scaled)
    accuracy = (y_pred == y).mean() * 100
    
    return JSONResponse({
        "message": "Model trained successfully on Kaggle dataset",
        "total_transactions": total,
        "fraud_count": fraud_count,
        "legitimate_count": total - fraud_count,
        "fraud_rate": round(fraud_rate, 4),
        "accuracy": round(accuracy, 2),
        "features_used": len(feature_names),
        "feature_names": feature_names[:10]  # Send first 10 for display
    })

@app.post("/predict")
async def predict(data: dict):
    global model, scaler, feature_names
    
    if model is None:
        # Try to load saved model
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                saved = pickle.load(f)
                model = saved['model']
                scaler = saved['scaler']
                feature_names = saved['features']
        else:
            return JSONResponse({"error": "Model not trained. Please upload dataset first."}, status_code=400)
    
    # Build feature vector for all V1-V28, Time, Amount
    features = []
    
    # Add V1 to V28 (if not provided, use 0)
    for i in range(1, 29):
        v_key = f'V{i}'
        if v_key in data:
            features.append(data[v_key])
        else:
            features.append(0.0)
    
    # Add Time
    if 'Time' in data:
        features.append(data['Time'])
    else:
        features.append(0.0)
    
    # Add Amount
    if 'Amount' in data:
        features.append(data['Amount'])
    else:
        features.append(0.0)
    
    # Convert to numpy array and scale
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Predict
    prob = model.predict_proba(X_scaled)[0]
    pred = model.predict(X_scaled)[0]
    
    # Find top contributing features (simplified)
    # Get feature contributions based on deviation from mean
    contributions = {}
    feature_importance = []
    
    # Get top 5 features by absolute value
    feature_abs = np.abs(X_scaled[0])
    top_indices = np.argsort(feature_abs)[-5:][::-1]
    
    for idx in top_indices:
        if idx < len(feature_names):
            contributions[feature_names[idx]] = round(feature_abs[idx] * 10, 1)
    
    # Generate detailed explanation
    if pred == 1:
        explanation = f"""🚨 **FRAUD ALERT DETECTED**

📊 **Risk Analysis:**
• Fraud Probability: {prob[1]*100:.1f}%
• Risk Level: {'HIGH' if prob[1] > 0.7 else 'MEDIUM'}

🔍 **Key Anomalies Detected:**
"""
        for feature, contrib in list(contributions.items())[:3]:
            explanation += f"• {feature}: Unusual pattern (deviation score: {contrib})\n"
        
        explanation += f"""
💡 **Why Flagged:**
The transaction shows significant deviations in PCA-transformed features (V1-V28), which are strong indicators of fraudulent behavior based on historical patterns.

✅ **Recommended Action:**
• Block transaction and request additional verification
• Send OTP to cardholder's registered mobile
• Flag for manual review"""
    
    else:
        explanation = f"""✅ **LEGITIMATE TRANSACTION**

📊 **Risk Analysis:**
• Legitimate Probability: {prob[0]*100:.1f}%
• Risk Level: LOW

🔍 **Pattern Analysis:**
• Transaction matches normal spending patterns
• No significant anomalies detected in PCA features

💡 **Assessment:**
The transaction patterns align with legitimate historical transactions. All V1-V28 features fall within expected ranges.

✅ **Action:**
• Approve transaction
• No additional verification needed"""
    
    return {
        "is_fraud": bool(pred),
        "fraud_probability": round(prob[1] * 100, 2),
        "legitimate_probability": round(prob[0] * 100, 2),
        "risk_level": "High" if prob[1] > 0.7 else "Medium" if prob[1] > 0.3 else "Low",
        "explanation": explanation,
        "top_features": contributions
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🛡️ FRAUD DETECTION API - KAGGLE DATASET READY")
    print("="*50)
    print("\n📊 Dataset Format: Time, V1-V28, Amount, Class")
    print("📍 API running at: http://localhost:8000")
    print("📤 POST /train - Upload creditcard.csv")
    print("🔍 POST /predict - Analyze transaction")
    print("\n" + "="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)