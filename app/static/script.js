// Global variables
let chart = null;
let currentPrediction = null;
let recentPredictions = [];

// DOM Elements
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const loadingState = document.getElementById('loadingState');
const resultsState = document.getElementById('resultsState');
const initialState = document.getElementById('initialState');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeDragAndDrop();
    initializeEventListeners();
    loadRecentPredictions();
});

// Drag and Drop functionality
function initializeDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    dropArea.classList.add('drag-over');
}

function unhighlight() {
    dropArea.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Process uploaded file
async function processFile(file) {
    // Validate file
    if (!validateFile(file)) {
        showError('Invalid file type. Please upload an image (JPG, PNG, BMP, TIFF).');
        return;
    }
    
    // Show loading state
    showLoadingState();
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Send to server
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Store prediction
            currentPrediction = data;
            
            // Update UI with results
            updateResultsUI(data);
            
            // Save to recent predictions
            saveRecentPrediction(data);
            
            // Show results
            showResultsState();
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Connection error. Please try again.');
    }
}

// Validate file
function validateFile(file) {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!allowedTypes.includes(file.type)) {
        return false;
    }
    
    if (file.size > maxSize) {
        return false;
    }
    
    return true;
}

// Update UI with results
function updateResultsUI(data) {
    // Update timestamp
    document.getElementById('timestamp').textContent = data.prediction.timestamp;
    
    // Update prediction
    document.getElementById('predictionText').textContent = data.class_info.name;
    document.getElementById('predictionAbbr').textContent = `(${data.class_info.abbr})`;
    document.getElementById('confidenceText').textContent = data.prediction.confidence;
    
    // Update severity
    const severity = data.class_info.severity.toLowerCase();
    const severityDot = document.querySelector('.severity-dot');
    const severityText = document.getElementById('severityText');
    
    severityDot.className = 'severity-dot ' + severity;
    severityText.className = 'severity-text ' + severity;
    severityText.textContent = severity.toUpperCase();
    
    // Update images
    document.getElementById('originalImage').src = 'data:image/png;base64,' + data.display_image;
    document.getElementById('heatmapImage').src = 'data:image/png;base64,' + data.heatmap;
    
    // Update medical info
    updateMedicalInfo(data.class_info);
    
    // Update recommendations
    updateRecommendations(data.class_info);
    
    // Create chart
    createChart(data.sorted_probabilities);
}

// Update medical information
function updateMedicalInfo(classInfo) {
    const infoGrid = document.getElementById('medicalInfo');
    
    const infoItems = [
        {
            title: 'Condition',
            content: classInfo.name,
            icon: 'fas fa-disease'
        },
        {
            title: 'Severity Level',
            content: classInfo.severity,
            icon: 'fas fa-exclamation-triangle'
        },
        {
            title: 'Description',
            content: classInfo.description,
            icon: 'fas fa-info-circle'
        },
        {
            title: 'Common Locations',
            content: classInfo.common_locations,
            icon: 'fas fa-map-marker-alt'
        },
        {
            title: 'Recommended Action',
            content: classInfo.action,
            icon: 'fas fa-clipboard-check'
        },
        {
            title: 'Accuracy',
            content: '84.3% (Test Set)',
            icon: 'fas fa-chart-line'
        }
    ];
    
    infoGrid.innerHTML = infoItems.map(item => `
        <div class="info-item slide-in">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <i class="${item.icon}" style="color: var(--primary);"></i>
                <h4>${item.title}</h4>
            </div>
            <p>${item.content}</p>
        </div>
    `).join('');
}

// Update recommendations
function updateRecommendations(classInfo) {
    const recommendationsDiv = document.getElementById('recommendations');
    
    const recommendations = [
        {
            text: classInfo.action,
            icon: 'fas fa-user-md'
        },
        {
            text: 'Monitor lesion for changes in size, shape, or color',
            icon: 'fas fa-eye'
        },
        {
            text: 'Take clear photos every month for comparison',
            icon: 'fas fa-camera'
        },
        {
            text: 'Protect skin from UV exposure with SPF 30+ sunscreen',
            icon: 'fas fa-sun'
        },
        {
            text: 'Regular skin examinations by a professional',
            icon: 'fas fa-stethoscope'
        }
    ];
    
    recommendationsDiv.innerHTML = `
        <ul>
            ${recommendations.map(rec => `
                <li>
                    <i class="${rec.icon}"></i>
                    ${rec.text}
                </li>
            `).join('')}
        </ul>
    `;
}

// Create chart
function createChart(probabilities) {
    const ctx = document.getElementById('probabilitiesChart').getContext('2d');
    
    // Destroy existing chart
    if (chart) {
        chart.destroy();
    }
    
    // Prepare data
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities);
    const colors = generateColors(labels.length);
    
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.6', '1')),
                borderWidth: 2,
                borderRadius: 6,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        drawBorder: false
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

// Generate colors for chart
function generateColors(count) {
    const colors = [
        'rgba(59, 130, 246, 0.6)',  // Blue
        'rgba(139, 92, 246, 0.6)',  // Purple
        'rgba(16, 185, 129, 0.6)',  // Green
        'rgba(245, 158, 11, 0.6)',  // Yellow
        'rgba(239, 68, 68, 0.6)',   // Red
        'rgba(14, 165, 233, 0.6)',  // Sky
        'rgba(244, 63, 94, 0.6)'    // Pink
    ];
    
    return colors.slice(0, count);
}

// State management
function showLoadingState() {
    initialState.style.display = 'none';
    resultsState.style.display = 'none';
    loadingState.style.display = 'block';
    
    // Animate loading steps
    const steps = document.querySelectorAll('.step');
    let currentStep = 0;
    
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            currentStep++;
        } else {
            clearInterval(stepInterval);
        }
    }, 500);
}

function showResultsState() {
    loadingState.style.display = 'none';
    resultsState.style.display = 'block';
    
    // Show recent predictions section
    const recentSection = document.getElementById('recentPredictions');
    if (recentPredictions.length > 0) {
        recentSection.style.display = 'block';
    }
}

function resetAnalysis() {
    resultsState.style.display = 'none';
    initialState.style.display = 'block';
    fileInput.value = '';
    
    // Reset loading steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active');
    });
}

// Recent predictions
function saveRecentPrediction(data) {
    const prediction = {
        id: Date.now(),
        className: data.class_info.name,
        classAbbr: data.class_info.abbr,
        confidence: data.prediction.raw_confidence,
        severity: data.class_info.severity,
        timestamp: data.prediction.timestamp,
        data: data
    };
    
    // Add to beginning of array
    recentPredictions.unshift(prediction);
    
    // Keep only last 5
    if (recentPredictions.length > 5) {
        recentPredictions.pop();
    }
    
    // Save to localStorage
    localStorage.setItem('recentPredictions', JSON.stringify(recentPredictions));
    
    // Update UI
    updateRecentPredictionsUI();
}

function loadRecentPredictions() {
    const saved = localStorage.getItem('recentPredictions');
    if (saved) {
        recentPredictions = JSON.parse(saved);
        updateRecentPredictionsUI();
    }
}

function updateRecentPredictionsUI() {
    const recentList = document.getElementById('recentList');
    const recentSection = document.getElementById('recentPredictions');
    
    if (recentPredictions.length === 0) {
        recentSection.style.display = 'none';
        return;
    }
    
    recentSection.style.display = 'block';
    
    recentList.innerHTML = recentPredictions.map(pred => `
        <div class="recent-item" onclick="loadRecent(${pred.id})">
            <div class="recent-info">
                <div class="recent-class">${pred.className} (${pred.classAbbr})</div>
                <div class="recent-time">${pred.timestamp}</div>
            </div>
            <div class="recent-confidence ${getSeverityClass(pred.severity)}">
                ${pred.confidence.toFixed(1)}%
            </div>
        </div>
    `).join('');
}

function getSeverityClass(severity) {
    switch (severity.toLowerCase()) {
        case 'high': return 'text-danger';
        case 'medium': return 'text-warning';
        case 'low': return 'text-success';
        default: return 'text-info';
    }
}

function loadRecent(id) {
    const prediction = recentPredictions.find(p => p.id === id);
    if (prediction) {
        currentPrediction = prediction.data;
        updateResultsUI(prediction.data);
        showResultsState();
        
        // Highlight selected item
        document.querySelectorAll('.recent-item').forEach(item => {
            item.classList.remove('active');
        });
        event.currentTarget.classList.add('active');
    }
}

// Utility functions
function showError(message) {
    alert(`Error: ${message}`);
    resetAnalysis();
}

function scheduleAppointment() {
    alert('This would integrate with a calendar system in a production app.\nFor now, please contact a dermatologist directly.');
}

function saveReport() {
    if (!currentPrediction) {
        showError('No prediction to save');
        return;
    }
    
    // Create report content
    const report = `
        SKIN CANCER DIAGNOSIS REPORT
        =============================
        
        Date: ${currentPrediction.prediction.timestamp}
        
        DIAGNOSIS:
        ----------
        Condition: ${currentPrediction.class_info.name} (${currentPrediction.class_info.abbr})
        Confidence: ${currentPrediction.prediction.confidence}
        Severity: ${currentPrediction.class_info.severity}
        
        MEDICAL INFORMATION:
        -------------------
        Description: ${currentPrediction.class_info.description}
        Common Locations: ${currentPrediction.class_info.common_locations}
        
        RECOMMENDATIONS:
        ---------------
        1. ${currentPrediction.class_info.action}
        2. Monitor lesion for changes
        3. Regular professional examinations
        4. Sun protection with SPF 30+
        
        DISCLAIMER:
        ----------
        This report is for educational purposes only.
        Always consult a medical professional for diagnosis.
        
        Model Accuracy: 84.3%
        Sensitivity: 91%
        
        --- End of Report ---
    `;
    
    // Create download link
    const blob = new Blob([report], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `skin_diagnosis_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    alert('Report saved successfully!');
}

function printResults() {
    window.print();
}

// Initialize event listeners
function initializeEventListeners() {
    // Browse button
    document.querySelector('.browse-btn').addEventListener('click', () => {
        fileInput.click();
    });
    
    // Reset button
    document.querySelector('[onclick="resetAnalysis()"]').addEventListener('click', resetAnalysis);
    
    // Save report button
    document.querySelector('[onclick="saveReport()"]').addEventListener('click', saveReport);
    
    // Schedule appointment button
    document.querySelector('[onclick="scheduleAppointment()"]').addEventListener('click', scheduleAppointment);
    
    // Print button
    document.querySelector('[onclick="printResults()"]').addEventListener('click', printResults);
}

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl + O to open file dialog
    if (e.ctrlKey && e.key === 'o') {
        e.preventDefault();
        fileInput.click();
    }
    
    // Esc to reset
    if (e.key === 'Escape') {
        resetAnalysis();
    }
    
    // Ctrl + P to print
    if (e.ctrlKey && e.key === 'p') {
        e.preventDefault();
        printResults();
    }
});
