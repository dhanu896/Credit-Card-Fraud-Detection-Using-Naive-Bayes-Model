const API_URL = 'http://localhost:8000';

let modelTrained = false;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadStatus = document.getElementById('uploadStatus');
const metricsGrid = document.getElementById('metricsGrid');
const predictionCard = document.getElementById('predictionCard');
const predictBtn = document.getElementById('predictBtn');
const infoCard = document.getElementById('infoCard');

// Generate V1-V28 input fields
function generateVFields() {
    const container = document.getElementById('advancedFeatures');
    container.innerHTML = '';
    
    for (let i = 5; i <= 28; i++) {
        const div = document.createElement('div');
        div.className = 'form-group';
        div.innerHTML = `
            <label style="font-size: 11px;">V${i}</label>
            <input type="text" id="v${i}" placeholder="0.0" value="0">
        `;
        container.appendChild(div);
    }
}

generateVFields();

// File upload handler
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '#764ba2';
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = '#667eea';
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
        uploadFile(file);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        uploadFile(e.target.files[0]);
    }
});

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    uploadStatus.innerHTML = '<div class="status-message status-success">📤 Training Naive Bayes model on Kaggle dataset...</div>';
    
    try {
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            uploadStatus.innerHTML = `<div class="status-message status-success">
                ✅ ${data.message}<br>
                📊 Total: ${data.total_transactions.toLocaleString()} | Fraud: ${data.fraud_count} (${data.fraud_rate}%) | Accuracy: ${data.accuracy}%
            </div>`;
            
            // Update metrics
            document.getElementById('totalTx').innerText = data.total_transactions.toLocaleString();
            document.getElementById('fraudTx').innerText = data.fraud_count;
            document.getElementById('legitTx').innerText = data.legitimate_count.toLocaleString();
            document.getElementById('fraudRate').innerText = `${data.fraud_rate}%`;
            document.getElementById('accuracy').innerText = `${data.accuracy}%`;
            
            // Show feature info
            if (data.feature_names) {
                document.getElementById('featureList').innerHTML = `
                    <strong>Features used:</strong> ${data.feature_names.join(', ')}...
                    <br><small>Total ${data.features_used} PCA features</small>
                `;
            }
            
            // Show all sections
            metricsGrid.style.display = 'grid';
            predictionCard.style.display = 'block';
            infoCard.style.display = 'block';
            modelTrained = true;
            
        } else {
            uploadStatus.innerHTML = `<div class="status-message status-error">❌ Error: ${data.error || 'Upload failed'}</div>`;
        }
    } catch (error) {
        uploadStatus.innerHTML = `<div class="status-message status-error">❌ Connection error: Make sure backend is running on port 8000</div>`;
    }
}

// Prediction handler
predictBtn.addEventListener('click', async () => {
    if (!modelTrained) {
        alert('Please upload creditcard.csv and train the model first!');
        return;
    }
    
    // Collect all features
    const features = {
        Amount: parseFloat(document.getElementById('amount').value) || 0,
        Time: parseFloat(document.getElementById('time').value) || 0
    };
    
    // Add V1-V28
    for (let i = 1; i <= 28; i++) {
        const val = document.getElementById(`v${i}`).value;
        features[`V${i}`] = parseFloat(val) || 0;
    }
    
    // Show loading
    predictBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing with Naive Bayes...';
    predictBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(features)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPrediction(result);
        } else {
            alert('Prediction failed: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Connection error: Make sure backend is running');
    } finally {
        predictBtn.innerHTML = '<span class="btn-icon">🔮</span> Analyze Transaction with Naive Bayes';
        predictBtn.disabled = false;
    }
});

function displayPrediction(result) {
    const resultDiv = document.getElementById('predictionResult');
    const explanationDiv = document.getElementById('aiExplanation');
    
    const isFraud = result.is_fraud;
    const prob = isFraud ? result.fraud_probability : result.legitimate_probability;
    const risk = result.risk_level;
    
    resultDiv.innerHTML = `
        <div class="prediction-result ${isFraud ? 'result-fraud' : 'result-legit'}">
            <h3>${isFraud ? '🚨 FRAUD DETECTED!' : '✅ LEGITIMATE TRANSACTION'}</h3>
            <div class="probability">${prob.toFixed(1)}%</div>
            <div class="risk-badge risk-${risk.toLowerCase()}">${risk} Risk Level</div>
        </div>
    `;
    
    // Format explanation with markdown-like styling
    const formattedExplanation = result.explanation
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/•/g, '•')
        .replace(/\n/g, '<br>');
    
    explanationDiv.innerHTML = `
        <h4>🤖 Naive Bayes Model Explanation</h4>
        <div style="line-height: 1.8;">${formattedExplanation}</div>
        ${result.top_features ? `
        <hr style="margin: 15px 0;">
        <strong>📊 Top Anomalous PCA Features:</strong>
        <ul style="margin-top: 10px; margin-left: 20px;">
            ${Object.entries(result.top_features).map(([k,v]) => `<li>${k}: deviation score ${v}</li>`).join('')}
        </ul>
        ` : ''}
        <p style="margin-top: 15px; font-size: 11px; color: #888; text-align: center;">
            ⚡ Gaussian Naive Bayes trained on Kaggle Credit Card Fraud Dataset<br>
            Using V1-V28 PCA components + Time + Amount features
        </p>
    `;
}

// Check backend
async function checkBackend() {
    try {
        const response = await fetch(`${API_URL}/`);
        if (response.ok) {
            console.log('✅ Backend connected');
        }
    } catch (error) {
        console.warn('⚠️ Backend not running');
        uploadStatus.innerHTML = '<div class="status-message status-error">⚠️ Backend not running. Run: cd backend && python app.py</div>';
    }
}

checkBackend();