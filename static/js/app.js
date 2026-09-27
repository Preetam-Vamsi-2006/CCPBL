/* ============================================================================
   SENTIMENT ANALYSIS - MODERN JAVASCRIPT
   ============================================================================ */

const API_BASE_URL = '';

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Sentiment Analysis App Loaded');
    
    // Setup event listeners
    setupNavigation();
    setupFormListeners();
    checkAPIStatus();
    loadModelMetrics();
    
    // Check API status periodically
    setInterval(checkAPIStatus, 30000);
    setInterval(loadModelMetrics, 60000); // Refresh metrics every minute
});

// ============================================================================
// NAVIGATION & SECTIONS
// ============================================================================

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('data-section');
            showSection(section);
            
            // Update active state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.add('active');
        // Scroll to section
        setTimeout(() => {
            section.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
}

// ============================================================================
// FORM LISTENERS
// ============================================================================

function setupFormListeners() {
    // Character count for single review
    const reviewInput = document.getElementById('reviewInput');
    if (reviewInput) {
        reviewInput.addEventListener('input', (e) => {
            const count = e.target.value.length;
            const charCount = document.getElementById('charCount');
            if (charCount) {
                charCount.textContent = Math.min(count, 1000);
            }
        });
    }
    
    // Batch input counter
    const batchInput = document.getElementById('batchInput');
    if (batchInput) {
        batchInput.addEventListener('input', (e) => {
            const lines = e.target.value.split('\n').filter(line => line.trim().length > 0);
            const batchCharInfo = document.getElementById('batchCharInfo');
            if (batchCharInfo) {
                batchCharInfo.textContent = lines.length;
            }
        });
    }
}

// ============================================================================
// API STATUS CHECK
// ============================================================================

async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        const isHealthy = data.status === 'healthy' && data.model_loaded;
        updateStatusIndicator(isHealthy);
        
    } catch (error) {
        console.error('❌ Health check failed:', error);
        updateStatusIndicator(false);
    }
}

function updateStatusIndicator(isOnline) {
    // Navigation status
    const navStatus = document.querySelector('.nav-status');
    if (navStatus) {
        const statusDot = navStatus.querySelector('.status-dot');
        const statusText = navStatus.querySelector('.status-text');
        
        if (isOnline) {
            statusDot.classList.remove('offline');
            statusDot.classList.add('online');
            statusText.textContent = 'Online';
        } else {
            statusDot.classList.remove('online');
            statusDot.classList.add('offline');
            statusText.textContent = 'Offline';
        }
    }
    
    // Footer status
    const footerStatus = document.getElementById('footerStatus');
    if (footerStatus) {
        footerStatus.textContent = isOnline ? 'API Status: Online ✓' : 'API Status: Offline';
    }
}

// ============================================================================
// SINGLE REVIEW ANALYSIS
// ============================================================================

async function analyzeSingle(event) {
    event.preventDefault();
    
    const review = document.getElementById('reviewInput').value.trim();
    const resultPanel = document.getElementById('resultPanel');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const statusMessage = document.getElementById('statusMessage');
    
    // Validation
    if (review.length < 10) {
        showMessage('statusMessage', 'error', '❌ Review must be at least 10 characters long');
        return;
    }
    
    // Show loading
    if (loadingSpinner) loadingSpinner.style.display = 'flex';
    if (resultPanel) resultPanel.style.display = 'none';
    showMessage('statusMessage', 'loading', '⏳ Analyzing your review...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ review })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        // Display result
        displaySingleResult(data);
        showMessage('statusMessage', 'success', '✅ Analysis complete!');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showMessage('statusMessage', 'error', `❌ Error: ${error.message}`);
    } finally {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
    }
}

function displaySingleResult(data) {
    const resultPanel = document.getElementById('resultPanel');
    const badge = document.getElementById('sentimentBadge');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    const textPreview = document.getElementById('textPreview');
    const predictionValue = document.getElementById('predictionValue');
    const classificationValue = document.getElementById('classificationValue');
    
    // Update sentiment badge
    const sentiment = data.sentiment.toUpperCase();
    badge.textContent = sentiment;
    badge.className = `sentiment-badge ${data.sentiment}`;
    badge.innerHTML = `
        <i class="fas fa-${data.sentiment === 'positive' ? 'smile' : 'frown'}"></i>
        <span>${sentiment}</span>
    `;
    
    // Update confidence
    const confidence = data.confidence * 100;
    confidenceFill.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence.toFixed(2)}%`;
    
    // Update text preview
    textPreview.textContent = data.text_preview || 'N/A';
    
    // Update stats
    predictionValue.textContent = data.prediction;
    classificationValue.textContent = data.sentiment;
    
    // Show result panel
    resultPanel.style.display = 'block';
}

function closeResult() {
    document.getElementById('resultPanel').style.display = 'none';
    document.getElementById('statusMessage').innerHTML = '';
}

// ============================================================================
// BATCH ANALYSIS
// ============================================================================

async function analyzeBatch() {
    const batchInput = document.getElementById('batchInput').value.trim();
    const batchResultPanel = document.getElementById('batchResultPanel');
    const batchLoadingSpinner = document.getElementById('batchLoadingSpinner');
    const batchStatusMessage = document.getElementById('batchStatusMessage');
    
    // Parse reviews
    const reviews = batchInput
        .split('\n')
        .map(r => r.trim())
        .filter(r => r.length > 0);
    
    // Validation
    if (reviews.length === 0) {
        showMessage('batchStatusMessage', 'error', '❌ Please enter at least one review');
        return;
    }
    
    if (reviews.length > 100) {
        showMessage('batchStatusMessage', 'error', '❌ Maximum 100 reviews per batch');
        return;
    }
    
    // Show loading
    if (batchLoadingSpinner) batchLoadingSpinner.style.display = 'flex';
    if (batchResultPanel) batchResultPanel.style.display = 'none';
    showMessage('batchStatusMessage', 'loading', `⏳ Analyzing ${reviews.length} reviews...`);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/batch-analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ reviews })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Batch analysis failed');
        }
        
        // Display results
        displayBatchResults(data);
        showMessage('batchStatusMessage', 'success', '✅ Batch analysis complete!');
        
    } catch (error) {
        console.error('Batch analysis error:', error);
        showMessage('batchStatusMessage', 'error', `❌ Error: ${error.message}`);
    } finally {
        if (batchLoadingSpinner) batchLoadingSpinner.style.display = 'none';
    }
}

function displayBatchResults(data) {
    const batchResultPanel = document.getElementById('batchResultPanel');
    
    // Update stats
    document.getElementById('totalReviews').textContent = data.total;
    document.getElementById('positiveCount').textContent = data.positive_count;
    document.getElementById('negativeCount').textContent = data.negative_count;
    document.getElementById('avgConfidence').textContent = (data.average_confidence * 100).toFixed(2) + '%';
    
    // Update table
    const tableBody = document.getElementById('resultsTableBody');
    tableBody.innerHTML = '';
    
    data.results.forEach((result, index) => {
        const row = document.createElement('tr');
        const sentiment = result.sentiment ? result.sentiment.toUpperCase() : 'ERROR';
        const confidence = result.confidence ? (result.confidence * 100).toFixed(2) : 0;
        const icon = result.sentiment === 'positive' ? '😊' : '😞';
        
        row.innerHTML = `
            <td><strong>#${index + 1}</strong></td>
            <td>${escapeHtml(result.text_preview || 'N/A')}</td>
            <td>
                <span style="display: inline-flex; align-items: center; gap: 8px;">
                    <span>${icon}</span>
                    <strong>${sentiment}</strong>
                </span>
            </td>
            <td><strong>${confidence}%</strong></td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Show result panel
    batchResultPanel.style.display = 'block';
}

function clearBatch() {
    document.getElementById('batchInput').value = '';
    document.getElementById('batchResultPanel').style.display = 'none';
    document.getElementById('batchStatusMessage').innerHTML = '';
    document.getElementById('batchCharInfo').textContent = '0';
}

function closeBatchResult() {
    document.getElementById('batchResultPanel').style.display = 'none';
    document.getElementById('batchStatusMessage').innerHTML = '';
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showMessage(elementId, type, message) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.className = `status-message show ${type}`;
    
    if (type === 'loading') {
        element.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner">
                    <div class="dot dot-1"></div>
                    <div class="dot dot-2"></div>
                    <div class="dot dot-3"></div>
                </div>
                <p>${message}</p>
            </div>
        `;
    } else {
        element.textContent = message;
    }
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// ============================================================================
// SMOOTH SCROLLING
// ============================================================================

document.addEventListener('click', (e) => {
    if (e.target.tagName === 'A' && e.target.href.includes('#')) {
        const hash = e.target.getAttribute('href');
        if (hash === '#') return;
        
        e.preventDefault();
        const section = hash.substring(1);
        showSection(section);
    }
});

// ============================================================================
// CONSOLE MESSAGES
// ============================================================================

console.log('%c🎬 Sentiment Analysis API', 'color: #667eea; font-size: 16px; font-weight: bold;');
console.log('%cAPI Endpoints:', 'color: #764ba2; font-weight: bold;');
console.log('  POST /api/analyze - Single review analysis');
console.log('  POST /api/batch-analyze - Batch analysis');
console.log('  GET /api/model-info - Model information');
console.log('  GET /health - Health check');


// ============================================================================
// MODEL METRICS
// ============================================================================

async function loadModelMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model-info`);
        const metrics = await response.json();
        
        if (response.ok) {
            displayModelMetrics(metrics);
        }
    } catch (error) {
        console.error('❌ Failed to load metrics:', error);
    }
}

function displayModelMetrics(metrics) {
    const container = document.getElementById('metricsContainer');
    if (!container) return;
    
    let metricsHTML = '';
    
    // Create metric cards
    if (metrics.accuracy !== undefined) {
        metricsHTML += createMetricCard('Accuracy', metrics.accuracy, 'accuracy');
    }
    if (metrics.precision !== undefined) {
        metricsHTML += createMetricCard('Precision', metrics.precision, 'precision');
    }
    if (metrics.recall !== undefined) {
        metricsHTML += createMetricCard('Recall', metrics.recall, 'recall');
    }
    if (metrics.f1_score !== undefined) {
        metricsHTML += createMetricCard('F1-Score', metrics.f1_score, 'f1');
    }
    
    // Add model information section
    if (metrics.model_type || metrics.features || metrics.dataset) {
        metricsHTML += createModelInfoSection(metrics);
    }
    
    container.innerHTML = metricsHTML;
}

function createMetricCard(name, value, type) {
    const percentage = (value * 100).toFixed(2);
    const icon = getMetricIcon(type);
    
    return `
        <div class="metric-card">
            <div class="metric-name">
                <i class="fas fa-${icon}"></i> ${name}
            </div>
            <div class="metric-value">${percentage}%</div>
            <div class="metric-bar">
                <div class="metric-bar-fill" style="width: ${percentage}%"></div>
            </div>
            <div style="font-size: 0.85em; color: #999; margin-top: 8px;">
                Raw Value: ${value.toFixed(4)}
            </div>
        </div>
    `;
}

function getMetricIcon(type) {
    const icons = {
        'accuracy': 'bullseye',
        'precision': 'crosshairs',
        'recall': 'magnifying-glass',
        'f1': 'balance-scale'
    };
    return icons[type] || 'chart-line';
}

function createModelInfoSection(metrics) {
    let html = '<div class="metrics-info">';
    html += '<h3>Model Information</h3>';
    html += '<div class="info-grid">';
    
    if (metrics.model_type) {
        html += `
            <div class="info-item">
                <div class="info-label">Model Type</div>
                <div class="info-value">${metrics.model_type}</div>
            </div>
        `;
    }
    
    if (metrics.version) {
        html += `
            <div class="info-item">
                <div class="info-label">Version</div>
                <div class="info-value">v${metrics.version}</div>
            </div>
        `;
    }
    
    if (metrics.features) {
        html += `
            <div class="info-item">
                <div class="info-label">Max Features</div>
                <div class="info-value">${metrics.features.max_features}</div>
            </div>
            <div class="info-item">
                <div class="info-label">N-gram Range</div>
                <div class="info-value">${metrics.features.ngram_range.join('-')}</div>
            </div>
        `;
    }
    
    if (metrics.dataset) {
        html += `
            <div class="info-item">
                <div class="info-label">Dataset</div>
                <div class="info-value">${metrics.dataset.name}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Total Reviews</div>
                <div class="info-value">${metrics.dataset.total_reviews.toLocaleString()}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Train/Test Split</div>
                <div class="info-value">${(metrics.dataset.train_split * 100).toFixed(0)}%/${(metrics.dataset.test_split * 100).toFixed(0)}%</div>
            </div>
            <div class="info-item">
                <div class="info-label">Classes</div>
                <div class="info-value">${metrics.dataset.classes.join(', ')}</div>
            </div>
        `;
    }
    
    html += '</div>';
    
    if (metrics.timestamp) {
        html += `
            <div class="timestamp">
                <strong>Last Updated:</strong> ${new Date(metrics.timestamp).toLocaleString()}
            </div>
        `;
    }
    
    html += '</div>';
    
    return html;
}
