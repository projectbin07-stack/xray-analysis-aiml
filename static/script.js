let predictionChart;
let accuracyChart;
let lossChart;

document.addEventListener('DOMContentLoaded', () => {
    initPerformanceCharts();
    setupEventListeners();
});

function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(name => {
        dropZone.addEventListener(name, e => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    dropZone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    document.querySelectorAll('.btn-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const previewImage = document.getElementById('previewImage');
            previewImage.style.filter = btn.dataset.view === 'analysis' 
                ? 'contrast(1.4) brightness(1.1) saturate(0) sepia(0.2)' 
                : 'none';
        });
    });
}

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    const previewImage = document.getElementById('previewImage');
    const previewWrapper = document.getElementById('previewWrapper');
    const imagePlaceholder = document.getElementById('imagePlaceholder');
    const scanTimestamp = document.getElementById('scanTimestamp');

    previewImage.src = URL.createObjectURL(file);
    previewWrapper.classList.remove('hidden');
    imagePlaceholder.classList.add('hidden');
    scanTimestamp.innerText = 'Scanned: ' + new Date().toLocaleTimeString();

    analyzeXray(file);
}

async function analyzeXray(file) {
    const loading = document.getElementById('loadingOverlay');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');

    loading.classList.remove('hidden');
    
    // Simulate steps for UX
    setTimeout(() => { step1.classList.add('dim'); step2.classList.remove('dim'); }, 800);
    setTimeout(() => { step2.classList.add('dim'); step3.classList.remove('dim'); }, 1600);

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok) {
            updateDashboard(data);
        } else {
            alert('Analysis Error: ' + data.error);
        }
    } catch (err) {
        console.error(err);
        alert('Diagnostic System Offline');
    } finally {
        setTimeout(() => {
            loading.classList.add('hidden');
            step1.classList.remove('dim');
            step2.classList.add('dim');
            step3.classList.add('dim');
        }, 2000);
    }
}

function updateDashboard(data) {
    const topPred = data.predictions[0];
    const riskScore = topPred.confidence;
    
    // Risk Card
    const riskValue = document.getElementById('riskValue');
    const statusBadge = document.getElementById('statusBadge');
    const riskProgress = document.getElementById('riskProgress');
    const riskExplanation = document.getElementById('riskExplanation');
    const metricsPanel = document.querySelector('.metrics-panel');

    riskValue.innerText = riskScore;
    statusBadge.innerText = data.status;
    riskProgress.style.width = riskScore + '%';

    metricsPanel.classList.remove('critical', 'attention', 'normal');
    if (data.status === 'CRITICAL') {
        metricsPanel.classList.add('critical');
        riskExplanation.innerText = "High-confidence detection of critical pathology. Immediate clinical intervention required.";
    } else if (data.status === 'NEEDS ATTENTION') {
        metricsPanel.classList.add('attention');
        riskExplanation.innerText = "Secondary indicators detected. Correlation with patient history and symptoms advised.";
    } else {
        metricsPanel.classList.add('normal');
        riskExplanation.innerText = "Neural analysis indicates physiological baseline within normal radiological limits.";
    }

    // Analytics Card
    document.getElementById('dominantCond').innerText = topPred.disease;
    document.getElementById('confConsistency').innerText = riskScore > 85 ? "High" : riskScore > 60 ? "Moderate" : "Low";
    document.getElementById('mAbnormality').innerText = (riskScore / 10).toFixed(1);
    document.getElementById('aSummaryText').innerText = `AI model has ${riskScore}% confidence in ${topPred.disease} detection. Findings are consistent with radiological patterns.`;

    // Predictions List & Chart
    updatePredictionsUI(data);
    updateChart(data.labels, data.confidence_scores);

    // AI Explanation
    document.getElementById('expSummary').innerText = data.explanation.summary;
    document.getElementById('expMeaning').innerText = data.explanation.meaning;
    document.getElementById('expRecommendation').innerText = data.explanation.recommendation;
}

function updatePredictionsUI(data) {
    const list = document.getElementById('topPredList');
    list.innerHTML = '';

    data.predictions.slice(0, 5).forEach((p, index) => {
        const row = document.createElement('div');
        row.className = `pred-row ${index < 3 ? 'highlight' : ''}`;
        row.innerHTML = `
            <div class="pred-info">
                <span class="pred-name">${p.disease}</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill" style="width: ${p.confidence}%; background-color: ${getBarColor(p.confidence)}"></div>
                </div>
            </div>
            <span class="pred-perc">${p.confidence}%</span>
        `;
        list.appendChild(row);
    });
}

function getBarColor(score) {
    if (score > 70) return '#f85149';
    if (score > 40) return '#d29922';
    return '#2f81f7';
}

function updateChart(labels, scores) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    if (predictionChart) predictionChart.destroy();

    predictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: scores,
                backgroundColor: scores.map(s => getBarColor(s)),
                borderRadius: 4,
                barThickness: 10
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { beginAtZero: true, max: 100, grid: { color: '#30363d', drawBorder: false }, ticks: { color: '#8b949e', font: { size: 9 } } },
                y: { grid: { display: false }, ticks: { color: '#f0f6fc', font: { size: 9 } } }
            }
        }
    });
}

function initPerformanceCharts() {
    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    const ctxLoss = document.getElementById('lossChart').getContext('2d');

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { x: { display: false }, y: { display: false } },
        elements: { point: { radius: 0 }, line: { tension: 0.4, borderWidth: 2 } }
    };

    accuracyChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: [1,2,3,4,5,6,7],
            datasets: [{ data: [88, 90, 89, 92, 94, 93, 95], borderColor: '#3fb950', fill: true, backgroundColor: 'rgba(63, 185, 80, 0.05)' }]
        },
        options: commonOptions
    });

    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: [1,2,3,4,5,6,7],
            datasets: [{ data: [0.4, 0.35, 0.32, 0.25, 0.18, 0.15, 0.12], borderColor: '#f85149', fill: true, backgroundColor: 'rgba(248, 81, 73, 0.05)' }]
        },
        options: commonOptions
    });
}
