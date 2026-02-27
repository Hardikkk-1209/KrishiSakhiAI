// ══════════════════════════════════════════
// KrishiSakhiAI — Frontend Logic (app.js)
// ══════════════════════════════════════════

const API = '';  // same origin
let farmerProfile = JSON.parse(localStorage.getItem('farmerProfile') || 'null');
let chatHistory = [];
let vaxRecords = JSON.parse(localStorage.getItem('vaxRecords') || '[]');
let dietRecords = JSON.parse(localStorage.getItem('dietRecords') || '[]');
let farmInputRecords = JSON.parse(localStorage.getItem('farmInputRecords') || '[]');
let currentMode = 'chat';

// ══════════════════════════════════════════
// INIT
// ══════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
    await checkHealth();
    await loadModels();
    loadRegions();
    setDefaultDates();

    if (farmerProfile) {
        completeOnboarding();
    } else {
        showPage('pageRegister');
    }
});

async function checkHealth() {
    try {
        const r = await fetch(`${API}/api/health`);
        const data = await r.json();
        const el = document.getElementById('ollamaStatus');
        el.className = 'status-badge ' + (data.ollama ? 'ok' : 'err');
        el.innerHTML = data.ollama ? '🟢 Ollama Connected' : '🔴 Ollama Offline';
    } catch (e) {
        document.getElementById('ollamaStatus').className = 'status-badge err';
        document.getElementById('ollamaStatus').innerHTML = '🔴 Server Error';
    }
}

async function loadModels() {
    try {
        const r = await fetch(`${API}/api/models`);
        const data = await r.json();
        const sel = document.getElementById('modelSelect');
        // Sort: chat models first, embedding models last
        const chatModels = data.models.filter(m => !m.includes('embed') && !m.includes('nomic'));
        const embedModels = data.models.filter(m => m.includes('embed') || m.includes('nomic'));
        const sorted = [...chatModels, ...embedModels];
        sel.innerHTML = sorted.map(m => `<option value="${m}">${m}</option>`).join('');
        if (sorted.length > 0) document.getElementById('modelSelector').style.display = 'block';
    } catch (e) { }
}

async function loadRegions() {
    try {
        const r = await fetch(`${API}/api/regions`);
        const data = await r.json();
        document.getElementById('regRegion').innerHTML = data.regions.map(r => `<option value="${r}">${r}</option>`).join('');
    } catch (e) { }
}

function setDefaultDates() {
    const today = new Date().toISOString().split('T')[0];
    ['vaxDate', 'dietDate', 'fiDate'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = today;
    });
}

// ══════════════════════════════════════════
// PAGE NAVIGATION
// ══════════════════════════════════════════
function showPage(id) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(id).classList.add('active');
}

function goStep(step) {
    if (step === 1) showPage('pageRegister');
    else if (step === 2) { renderCropPlan(); showPage('pageCropPlan'); }
    else if (step === 3) { renderRareCrops(); showPage('pageRareCrops'); }
}

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-tab[data-mode]').forEach(t => t.classList.remove('active'));
    document.querySelector(`.mode-tab[data-mode="${mode}"]`).classList.add('active');
    showPage(mode === 'chat' ? 'pageChat' : 'pageLivestock');
}

// ══════════════════════════════════════════
// STEP 1: REGISTRATION
// ══════════════════════════════════════════
function submitRegistration() {
    const name = document.getElementById('regName').value.trim();
    const phone = document.getElementById('regPhone').value.trim();
    const land = parseFloat(document.getElementById('regLand').value);
    const water = document.getElementById('regWater').value;
    const crop = document.getElementById('regCrop').value.trim();
    const region = document.getElementById('regRegion').value;

    const errEl = document.getElementById('regError');
    if (!name || !phone) {
        errEl.textContent = '❌ Please fill in your Name and Phone Number';
        errEl.style.display = 'block'; return;
    }
    if (phone.length < 10) {
        errEl.textContent = '❌ Please enter a valid 10-digit phone number';
        errEl.style.display = 'block'; return;
    }
    errEl.style.display = 'none';

    farmerProfile = { name, phone, land_size: land, water_source: water, current_crop: crop, region };
    localStorage.setItem('farmerProfile', JSON.stringify(farmerProfile));
    goStep(2);
}

// ══════════════════════════════════════════
// STEP 2: CROP PLANNING
// ══════════════════════════════════════════
async function renderCropPlan() {
    const region = farmerProfile.region;
    try {
        const r = await fetch(`${API}/api/crops/${region}`);
        const data = await r.json();
        const container = document.getElementById('cropPlanContent');

        let html = `
            <div class="hero-box" style="background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(5,150,105,0.08));">
                <h2 style="font-size:1.8rem;">🌾 Crop Planning — ${region}</h2>
                <p>Welcome, <strong>${farmerProfile.name}</strong>! Here are crops for your region.</p>
            </div>
            <div class="region-info">
                <h4>📍 ${region} Region Profile</h4>
                <p><strong>Climate:</strong> ${data.climate}</p>
                <p><strong>Soil Type:</strong> ${data.soil}</p>
                <p><strong>Your Land:</strong> ${farmerProfile.land_size} acres | <strong>Water:</strong> ${farmerProfile.water_source}</p>
            </div>
            <div class="section-label">🌱 Recommended Crops</div>
            <div class="crop-grid">
                ${data.common.map(c => `<div class="crop-chip">🌿 ${c}</div>`).join('')}
            </div>
            <div class="section-label">✨ Rare / Exotic Crop Opportunities</div>
        `;

        data.rare.forEach(crop => {
            html += `
                <div class="rare-card">
                    <h4>🌟 ${crop}</h4>
                    <p>High-value crop rarely grown in ${region}. Explore disease management in the next step.</p>
                </div>
            `;
        });

        container.innerHTML = html;
    } catch (e) {
        document.getElementById('cropPlanContent').innerHTML = '<div class="alert alert-red">Failed to load crop data</div>';
    }
}

// ══════════════════════════════════════════
// STEP 3: RARE CROP DISEASES
// ══════════════════════════════════════════
async function renderRareCrops() {
    const region = farmerProfile.region;
    try {
        const cropData = await (await fetch(`${API}/api/crops/${region}`)).json();
        const container = document.getElementById('rareCropContent');

        let html = `
            <div class="hero-box" style="background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(5,150,105,0.08));">
                <h2 style="font-size:1.8rem;">🔬 Rare Crop Disease Guide — ${region}</h2>
                <p>Complete disease management for high-value crops in your region</p>
            </div>
        `;

        for (const crop of cropData.rare) {
            const diseaseResp = await fetch(`${API}/api/rare-crops/${crop}`);
            const diseaseData = await diseaseResp.json();

            html += `<h3 style="color:var(--yellow); font-size:1.5rem; margin:1.5rem 0 0.5rem;">🌟 ${crop}</h3>`;

            diseaseData.diseases.forEach(d => {
                html += `
                    <div class="disease-card">
                        <h5>🦠 ${d.name}</h5>
                        <div class="disease-section symptoms"><strong>⚠️ Symptoms:</strong>${d.symptoms}</div>
                        <div class="disease-section prevention"><strong>🛡️ Preventive Measures:</strong>${d.prevention}</div>
                        <div class="disease-section treatment"><strong>💊 Treatment:</strong>${d.treatment}</div>
                        <div class="disease-section pesticide"><strong>🧪 Pesticides / Fungicides:</strong>${d.pesticides}</div>
                    </div>
                `;
            });
        }

        container.innerHTML = html;
    } catch (e) {
        document.getElementById('rareCropContent').innerHTML = '<div class="alert alert-red">Failed to load disease data</div>';
    }
}

// ══════════════════════════════════════════
// COMPLETE ONBOARDING
// ══════════════════════════════════════════
function completeOnboarding() {
    // Show sidebar elements
    document.getElementById('modeSelector').style.display = 'block';
    document.getElementById('resetProfileBtn').style.display = 'block';

    // Show farmer card in sidebar
    if (farmerProfile) {
        document.getElementById('farmerSidebar').style.display = 'block';
        document.getElementById('farmerSidebar').innerHTML = `
            <div class="farmer-card">
                <div class="name">👨‍🌾 ${farmerProfile.name}</div>
                <div class="meta">📍 ${farmerProfile.region} • ${farmerProfile.land_size} acres</div>
            </div>
        `;
    }

    setMode('chat');
    loadLivestockData();
}

function resetProfile() {
    localStorage.removeItem('farmerProfile');
    farmerProfile = null;
    chatHistory = [];
    showPage('pageRegister');
    document.getElementById('modeSelector').style.display = 'none';
    document.getElementById('resetProfileBtn').style.display = 'none';
    document.getElementById('farmerSidebar').style.display = 'none';
    document.getElementById('chatMessages').innerHTML = `
        <div class="welcome-box">
            <h2>🙏 Namaste! Welcome to KrishiSakhiAI!</h2>
            <p>I'm KrishiSakhi, your AI farming assistant.</p>
        </div>
    `;
}

function clearChat() {
    chatHistory = [];
    document.getElementById('chatMessages').innerHTML = `
        <div class="welcome-box">
            <h2>🙏 Namaste! Welcome to KrishiSakhiAI!</h2>
            <p>I'm KrishiSakhi, your AI farming assistant.</p>
        </div>
    `;
}

// ══════════════════════════════════════════
// CHAT
// ══════════════════════════════════════════
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';

    const messagesEl = document.getElementById('chatMessages');
    // Remove welcome box
    const wb = messagesEl.querySelector('.welcome-box');
    if (wb) wb.remove();

    // User message
    messagesEl.innerHTML += `
        <div class="msg user">
            <div class="msg-avatar">👤</div>
            <div class="msg-bubble">${escapeHtml(msg)}</div>
        </div>
    `;

    // AI placeholder
    const aiId = 'ai-' + Date.now();
    messagesEl.innerHTML += `
        <div class="msg ai" id="${aiId}">
            <div class="msg-avatar">🌾</div>
            <div class="msg-bubble"><span class="typing">Thinking...</span></div>
        </div>
    `;
    messagesEl.scrollTop = messagesEl.scrollHeight;

    const sendBtn = document.getElementById('chatSendBtn');
    sendBtn.disabled = true;

    try {
        const model = document.getElementById('modelSelect').value || 'llama3.2:1b';
        const temp = parseFloat(document.getElementById('temperature').value);

        const response = await fetch(`${API}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                model: model,
                temperature: temp,
                history: chatHistory.slice(-10),
                farmer_profile: farmerProfile
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        const bubble = document.querySelector(`#${aiId} .msg-bubble`);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(l => l.startsWith('data: '));

            for (const line of lines) {
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.token) {
                        fullResponse += data.token;
                        bubble.innerHTML = formatMarkdown(fullResponse);
                        messagesEl.scrollTop = messagesEl.scrollHeight;
                    }
                    if (data.error) {
                        bubble.innerHTML = `<span style="color:var(--red);">❌ ${data.error}</span>`;
                    }
                } catch (e) { }
            }
        }

        chatHistory.push({ role: 'user', content: msg });
        chatHistory.push({ role: 'assistant', content: fullResponse });

    } catch (e) {
        document.querySelector(`#${aiId} .msg-bubble`).innerHTML =
            `<span style="color:var(--red);">❌ Failed to connect. Is Ollama running?</span>`;
    }

    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    // Simple markdown formatting
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code style="background:rgba(16,185,129,0.15); padding:0.1rem 0.4rem; border-radius:4px; font-family:JetBrains Mono,monospace; font-size:0.85rem;">$1</code>')
        .replace(/\n/g, '<br>');
}

// ══════════════════════════════════════════
// LIVESTOCK — TAB SWITCHING
// ══════════════════════════════════════════
function switchLsTab(idx) {
    document.querySelectorAll('.ls-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
    document.querySelectorAll('.ls-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
}

function switchScanMode(mode, btn) {
    document.getElementById('scanManual').style.display = mode === 'manual' ? 'block' : 'none';
    document.getElementById('scanCsv').style.display = mode === 'csv' ? 'block' : 'none';
    btn.parentElement.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
}

// ══════════════════════════════════════════
// LIVESTOCK — HEALTH SCAN
// ══════════════════════════════════════════
async function runScan() {
    const fields = ['body_temp', 'heart_rate', 'respiratory_rate', 'activity_level', 'rumination_min',
        'feed_intake', 'water_intake', 'milk_yield', 'lying_time', 'steps_count',
        'gait_score', 'stance_symmetry', 'stride_length', 'ambient_temp', 'humidity_pct'];

    const reading = {};
    fields.forEach(f => { reading[f] = parseFloat(document.getElementById('s_' + f).value); });

    document.getElementById('scanResults').innerHTML = '<div class="alert alert-green">🔄 Running biosecurity scan...</div>';

    try {
        const r = await fetch(`${API}/api/livestock/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reading)
        });
        const data = await r.json();
        renderScanResults(data);
    } catch (e) {
        document.getElementById('scanResults').innerHTML = '<div class="alert alert-red">❌ Scan failed: ' + e.message + '</div>';
    }
}

function renderScanResults(data) {
    const healthColor = data.health.status_label === 'Healthy' ? 'green' : data.health.status_label === 'Stressed' ? 'orange' : 'red';
    const anomalyColor = data.anomaly.is_anomaly ? 'red' : 'green';
    const gaitColor = data.gait.gait_score < 2 ? 'green' : data.gait.gait_score < 3 ? 'orange' : 'red';

    document.getElementById('scanResults').innerHTML = `
        <div class="section-label" style="margin-top:1.5rem;">📊 Biosecurity Scan Results</div>
        <div class="result-grid">
            <div class="card card-${healthColor}">
                <div class="card-title">🏥 Health Status</div>
                <div class="card-value">${data.health.status_label}</div>
                <div class="card-sub">Confidence: ${(data.health.confidence * 100).toFixed(0)}%</div>
            </div>
            <div class="card card-${anomalyColor}">
                <div class="card-title">🔍 Anomaly Detection</div>
                <div class="card-value">${data.anomaly.is_anomaly ? '⚠️ Anomaly' : '✅ Normal'}</div>
                <div class="card-sub">Score: ${data.anomaly.anomaly_score?.toFixed(2) || 'N/A'}</div>
            </div>
            <div class="card card-${gaitColor}">
                <div class="card-title">🦶 Gait Analysis</div>
                <div class="card-value">${data.gait.gait_score?.toFixed(1) || 'N/A'}</div>
                <div class="card-sub">${data.gait.gait_label || 'N/A'}</div>
            </div>
            <div class="card">
                <div class="card-title">🔮 Disease Forecast</div>
                <div class="card-value" style="font-size:1.2rem; color:var(--yellow);">${(data.disease.predicted_disease || '').replace(/_/g, ' ')}</div>
                <div class="card-sub">Risk: ${(data.disease.confidence * 100)?.toFixed(0) || '?'}%</div>
            </div>
        </div>
        ${data.gait_cv ? `
        <div class="card" style="margin-top:1rem;">
            <div class="card-title">🎥 Computer Vision — Gait Analysis</div>
            <p style="color:var(--text-sub);">Score: ${data.gait_cv.gait_score?.toFixed(1)} | ${data.gait_cv.gait_label} | Symmetry: ${(data.gait_cv.symmetry_score * 100)?.toFixed(0)}%</p>
        </div>` : ''}
        ${data.behavior ? `
        <div class="card" style="margin-top:0.5rem;">
            <div class="card-title">🧠 Behavior Analysis</div>
            <p style="color:var(--text-sub);">${data.behavior.status || 'N/A'} | Alerts: ${data.behavior.alerts?.join(', ') || 'None'}</p>
        </div>` : ''}
    `;
}

async function uploadCsv() {
    const file = document.getElementById('csvFile').files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('scanResults').innerHTML = '<div class="alert alert-green">🔄 Processing CSV...</div>';
    try {
        const r = await fetch(`${API}/api/livestock/scan-csv`, { method: 'POST', body: formData });
        const data = await r.json();
        let html = `<div class="section-label">📊 Batch Scan Results (${data.total} animals)</div><table class="data-table"><thead><tr><th>#</th><th>Animal</th><th>Health</th><th>Confidence</th><th>Disease Risk</th><th>Gait</th><th>Anomaly</th></tr></thead><tbody>`;
        data.results.forEach(r => {
            html += `<tr><td>${r.id}</td><td>${r.animal_id}</td><td>${r.health}</td><td>${r.confidence}</td><td>${r.disease}</td><td>${r.gait}</td><td>${r.anomaly ? '⚠️' : '✅'}</td></tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('scanResults').innerHTML = html;
    } catch (e) {
        document.getElementById('scanResults').innerHTML = '<div class="alert alert-red">❌ CSV upload failed</div>';
    }
}

// ══════════════════════════════════════════
// LIVESTOCK — DATA TABLES & RECORDS
// ══════════════════════════════════════════
async function loadLivestockData() {
    // Vaccination schedule
    try {
        const r = await fetch(`${API}/api/reference/vaccination`);
        const data = await r.json();
        document.getElementById('vaxScheduleTable').innerHTML = buildTable(data.schedule, ['disease', 'vaccine', 'when', 'route']);
    } catch (e) { }

    // Diet reference
    try {
        const r = await fetch(`${API}/api/reference/diet`);
        const data = await r.json();
        document.getElementById('dietRefTable').innerHTML = buildTable(data.diet, ['component', 'qty', 'examples']);
    } catch (e) { }

    // MRL
    try {
        const r = await fetch(`${API}/api/reference/mrl`);
        const data = await r.json();
        document.getElementById('mrlTable').innerHTML = buildTable(data.guidelines, ['chemical', 'type', 'mrl', 'phi', 'risk']);
    } catch (e) { }

    // First Aid
    try {
        const r = await fetch(`${API}/api/reference/first-aid`);
        const data = await r.json();
        document.getElementById('firstAidTable').innerHTML = buildTable(data.first_aid, ['emergency', 'action', 'time']);
    } catch (e) { }

    // Vet Resources
    try {
        const r = await fetch(`${API}/api/reference/vet-resources`);
        const data = await r.json();
        document.getElementById('vetResourcesTable').innerHTML = buildTable(data.resources, ['service', 'find', 'coverage']);
    } catch (e) { }

    renderVaxRecords();
    renderDietRecords();
    renderFarmInputRecords();
}

function buildTable(rows, keys) {
    const headers = keys.map(k => k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, ' '));
    let html = '<table class="data-table"><thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    rows.forEach(row => {
        html += '<tr>' + keys.map(k => `<td>${row[k] || ''}</td>`).join('') + '</tr>';
    });
    return html + '</tbody></table>';
}

// ── Save Records ──
function saveVax() {
    const rec = {
        animal: document.getElementById('vaxAnimal').value,
        type: document.getElementById('vaxType').value,
        name: document.getElementById('vaxName').value,
        dose: document.getElementById('vaxDose').value,
        date: document.getElementById('vaxDate').value,
        vet: document.getElementById('vaxVet').value,
        notes: document.getElementById('vaxNotes').value,
        timestamp: new Date().toISOString()
    };
    vaxRecords.push(rec);
    localStorage.setItem('vaxRecords', JSON.stringify(vaxRecords));
    renderVaxRecords();
    // Clear form
    ['vaxAnimal', 'vaxName', 'vaxDose', 'vaxVet', 'vaxNotes'].forEach(id => document.getElementById(id).value = '');
}

function saveDiet() {
    const rec = {
        animal: document.getElementById('dietAnimal').value,
        date: document.getElementById('dietDate').value,
        lactation: document.getElementById('dietLact').value,
        green: document.getElementById('dietGreen').value,
        dry: document.getElementById('dietDry').value,
        concentrates: document.getElementById('dietConc').value,
        water: document.getElementById('dietWater').value,
        notes: document.getElementById('dietNotes').value,
        timestamp: new Date().toISOString()
    };
    dietRecords.push(rec);
    localStorage.setItem('dietRecords', JSON.stringify(dietRecords));
    renderDietRecords();
    ['dietAnimal', 'dietNotes'].forEach(id => document.getElementById(id).value = '');
}

function saveFarmInput() {
    const rec = {
        type: document.getElementById('fiType').value,
        name: document.getElementById('fiName').value,
        dose: document.getElementById('fiDose').value,
        field: document.getElementById('fiField').value,
        date: document.getElementById('fiDate').value,
        phi: document.getElementById('fiPhi').value,
        notes: document.getElementById('fiNotes').value,
        timestamp: new Date().toISOString()
    };
    farmInputRecords.push(rec);
    localStorage.setItem('farmInputRecords', JSON.stringify(farmInputRecords));
    renderFarmInputRecords();
    ['fiName', 'fiDose', 'fiField', 'fiNotes'].forEach(id => document.getElementById(id).value = '');
}

function renderVaxRecords() {
    if (vaxRecords.length === 0) { document.getElementById('vaxRecords').innerHTML = ''; return; }
    document.getElementById('vaxRecords').innerHTML =
        '<div class="section-label">📋 Saved Records (' + vaxRecords.length + ')</div>' +
        buildTable(vaxRecords, ['date', 'animal', 'type', 'name', 'dose', 'vet']);
}

function renderDietRecords() {
    if (dietRecords.length === 0) { document.getElementById('dietRecords').innerHTML = ''; return; }
    document.getElementById('dietRecords').innerHTML =
        '<div class="section-label">📋 Diet Logs (' + dietRecords.length + ')</div>' +
        buildTable(dietRecords, ['date', 'animal', 'lactation', 'green', 'dry', 'concentrates', 'water']);
}

function renderFarmInputRecords() {
    if (farmInputRecords.length === 0) { document.getElementById('farmInputRecords').innerHTML = ''; return; }
    document.getElementById('farmInputRecords').innerHTML =
        '<div class="section-label">📋 Application Records (' + farmInputRecords.length + ')</div>' +
        buildTable(farmInputRecords, ['date', 'type', 'name', 'dose', 'field', 'phi']);
}

// ── Emergency Vet ──
function findVet() {
    const location = document.getElementById('vetLocation').value.trim();
    if (!location) { alert('Please enter your location'); return; }
    const url = `https://www.google.com/maps/search/veterinary+hospital+near+${encodeURIComponent(location)}`;
    document.getElementById('vetResults').innerHTML = `
        <div class="card" style="margin-top:1rem;">
            <p style="color:var(--text-sub);">Searching for <strong>veterinary services</strong> near <strong>${escapeHtml(location)}</strong></p>
            <a href="${url}" target="_blank" class="btn btn-primary" style="margin-top:0.8rem; text-decoration:none;">🗺️ Open in Google Maps →</a>
        </div>
    `;
}
