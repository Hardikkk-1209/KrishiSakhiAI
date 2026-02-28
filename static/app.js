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
let selectedLang = localStorage.getItem('selectedLang') || 'English';
let pendingImage = null;

// ══════════════════════════════════════════
// INIT
// ══════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
    await checkHealth();
    await loadModels();
    loadRegions();
    setDefaultDates();

    // If profile exists → go to main app
    if (farmerProfile) {
        completeOnboarding();
    } else {
        // Otherwise show registration form
        showPage('pageRegister');
    }
});

async function checkHealth() {
    try {
        const r = await fetch(`${API}/api/health`);
        const data = await r.json();
        const el = document.getElementById('ollamaStatus');
        if (data.ollama) {
            el.className = 'status-badge ok';
            el.innerHTML = '<span class="dot dot-ok"></span>Ollama Connected';
        } else {
            el.className = 'status-badge err';
            el.innerHTML = '<span class="dot dot-err"></span>Ollama Offline';
        }
    } catch (e) {
        const el = document.getElementById('ollamaStatus');
        el.className = 'status-badge err';
        el.innerHTML = '<span class="dot dot-err"></span>Server Error';
    }
}

async function loadModels() {
    try {
        const r = await fetch(`${API}/api/models`);
        const data = await r.json();
        const sel = document.getElementById('modelSelect');
        const chatModels = data.models.filter(m => !m.includes('embed') && !m.includes('nomic'));
        const embedModels = data.models.filter(m => m.includes('embed') || m.includes('nomic'));
        const sorted = [...chatModels, ...embedModels];
        sel.innerHTML = sorted.map(m => `<option value="${m}">${m}</option>`).join('');
        if (sorted.length > 0) document.getElementById('modelSelector').style.display = 'block';
    } catch (e) {}
}

async function loadRegions() {
    try {
        const r = await fetch(`${API}/api/regions`);
        const data = await r.json();
        document.getElementById('regRegion').innerHTML =
            data.regions.map(r => `<option value="${r}">${r}</option>`).join('');
    } catch (e) {}
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
    const pages = document.querySelectorAll('.page');
    pages.forEach(p => p.classList.remove('active'));

    const target = document.getElementById(id);

    if (!target) {
        console.warn('Page not found:', id);
        return;
    }

    target.classList.add('active');
}

function goStep(step) {
    if (step === 1) showPage('pageRegister');
    else if (step === 2) { renderCropPlan(); showPage('pageCropPlan'); }
    else if (step === 3) { renderRareCrops(); showPage('pageRareCrops'); }
}

function setMode(mode) {
    currentMode = mode;
    // Update sidebar seg buttons
    document.querySelectorAll('.seg-btn[data-mode]').forEach(t => t.classList.remove('active'));
    const sb = document.querySelector(`.seg-btn[data-mode="${mode}"]`);
    if (sb) sb.classList.add('active');
    // Update pill tabs
    document.querySelectorAll('.pill-tab[data-mode]').forEach(t =>
        t.classList.toggle('active', t.dataset.mode === mode));
    // Update rail buttons
    document.querySelectorAll('.rail-btn[id^="rb-"]').forEach(b => b.classList.remove('active'));
    const rb = document.getElementById('rb-' + mode);
    if (rb) rb.classList.add('active');
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
            <div class="ob-hero" style="margin-bottom:1.5rem;">
                <div class="ob-tag">🌾 ${region} Region</div>
                <h1 style="font-size:1.6rem;">Crop Planning for ${farmerProfile.name}</h1>
                <p>Personalized recommendations based on your region and farm profile</p>
            </div>
            <div class="region-info">
                <h4>📍 ${region} Profile</h4>
                <p><strong>Climate:</strong> ${data.climate}</p>
                <p><strong>Soil Type:</strong> ${data.soil}</p>
                <p><strong>Your Land:</strong> ${farmerProfile.land_size} acres &nbsp;|&nbsp; <strong>Water:</strong> ${farmerProfile.water_source}</p>
            </div>
            <div class="section-label">🌱 Recommended Crops</div>
            <div class="crop-grid">
                ${data.common.map(c => `<div class="crop-chip">🌿 ${c}</div>`).join('')}
            </div>
            <div class="section-label">✨ High-Value Rare Crops</div>
        `;
        data.rare.forEach(crop => {
            html += `
                <div class="rare-card">
                    <h4>🌟 ${crop}</h4>
                    <p>High-value opportunity rarely grown in ${region}. Disease management details in the next step.</p>
                </div>`;
        });
        container.innerHTML = html;
    } catch (e) {
        document.getElementById('cropPlanContent').innerHTML =
            '<div class="alert alert-red">Failed to load crop data. Is the server running?</div>';
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
            <div class="ob-hero" style="margin-bottom:1.5rem;">
                <div class="ob-tag">🔬 Disease Management</div>
                <h1 style="font-size:1.6rem;">Rare Crop Disease Guide — ${region}</h1>
                <p>Complete disease management protocols for high-value crops in your area</p>
            </div>
        `;
        for (const crop of cropData.rare) {
            const diseaseResp = await fetch(`${API}/api/rare-crops/${crop}`);
            const diseaseData = await diseaseResp.json();
            html += `<h3 style="color:#92400e; font-size:1.3rem; margin:1.5rem 0 0.5rem; font-weight:800;">🌟 ${crop}</h3>`;
            diseaseData.diseases.forEach(d => {
                html += `
                    <div class="disease-card">
                        <h5>🦠 ${d.name}</h5>
                        <div class="disease-section symptoms"><strong>⚠️ Symptoms</strong>${d.symptoms}</div>
                        <div class="disease-section prevention"><strong>🛡️ Preventive Measures</strong>${d.prevention}</div>
                        <div class="disease-section treatment"><strong>💊 Treatment</strong>${d.treatment}</div>
                        <div class="disease-section pesticide"><strong>🧪 Pesticides / Fungicides</strong>${d.pesticides}</div>
                    </div>`;
            });
        }
        container.innerHTML = html;
    } catch (e) {
        document.getElementById('rareCropContent').innerHTML =
            '<div class="alert alert-red">Failed to load disease data.</div>';
    }
}

// ══════════════════════════════════════════
// COMPLETE ONBOARDING
// ══════════════════════════════════════════
function completeOnboarding() {
    const modeSel = document.getElementById('modeSelector');
    const resetBtn = document.getElementById('resetProfileBtn');
    const langSel = document.getElementById('langSelector');

    if (modeSel) modeSel.style.display = 'block';
    if (resetBtn) resetBtn.style.display = 'block';
    if (langSel) langSel.style.display = 'block';

    const langDropdown = document.getElementById('langSelect');
    if (langDropdown) langDropdown.value = selectedLang;

    if (farmerProfile) {
        const fSide = document.getElementById('farmerSidebar');
        if (fSide) {
            fSide.style.display = 'block';
            fSide.innerHTML = `
                <div class="farmer-card">
                    <div class="name">👨‍🌾 ${farmerProfile.name}</div>
                    <div class="meta">📍 ${farmerProfile.region} · ${farmerProfile.land_size} acres</div>
                </div>`;
        }

        const rv = document.getElementById('cspRegionVal');
        if (rv) rv.textContent = farmerProfile.region || '—';

        if (farmerProfile.region) {
            loadWeatherMini(farmerProfile.region);
        }
    }

    const month = new Date().getMonth();
    const seasons = ['Rabi','Rabi','Rabi','Summer','Summer','Kharif','Kharif','Kharif','Kharif','Rabi','Rabi','Rabi'];
    const sv = document.getElementById('cspSeasonVal');
    if (sv) sv.textContent = seasons[month] + ' Season';

    setMode('chat');
}

function resetProfile() {
    localStorage.removeItem('farmerProfile');
    farmerProfile = null;
    chatHistory = [];
    showPage('pageRegister');
    document.getElementById('modeSelector').style.display = 'none';
    document.getElementById('resetProfileBtn').style.display = 'none';
    document.getElementById('farmerSidebar').style.display = 'none';
    document.getElementById('langSelector').style.display = 'none';
    document.getElementById('chatMessages').innerHTML = `
        <div class="welcome-box">
            <div class="welcome-badge">🙏 Namaste!</div>
            <h2>Welcome to KrishiSakhiAI</h2>
            <p>Your AI-powered farming companion.</p>
        </div>`;
}

function clearChat() {
    chatHistory = [];
    document.getElementById('chatMessages').innerHTML = `
        <div class="welcome-box">
            <div class="welcome-badge">🙏 Namaste!</div>
            <h2>Welcome to KrishiSakhiAI</h2>
            <p>Your AI-powered farming companion. Ask me anything!</p>
            <div class="chip-row">
                <button class="chip" onclick="useChip(this)">🌱 Best kharif crops for my region?</button>
                <button class="chip" onclick="useChip(this)">🐛 How to control aphids organically?</button>
                <button class="chip" onclick="useChip(this)">💧 Drip irrigation setup guide</button>
                <button class="chip" onclick="useChip(this)">📈 Current MSP rates for wheat?</button>
            </div>
        </div>`;
}

function useChip(el) {
    document.getElementById('chatInput').value = el.textContent.trim();
    document.getElementById('chatInput').focus();
}

// ══════════════════════════════════════════
// CHAT
// ══════════════════════════════════════════
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg && !pendingImage) return;
    input.value = '';

    const messagesEl = document.getElementById('chatMessages');
    const wb = messagesEl.querySelector('.welcome-box');
    if (wb) wb.remove();

    if (pendingImage) {
        await analyzeImage(msg || 'What crop disease or pest do you see? Give diagnosis, treatment, and prevention.');
        return;
    }

    messagesEl.innerHTML += `
        <div class="msg user">
            <div class="msg-avatar">👤</div>
            <div class="msg-bubble">${escapeHtml(msg)}</div>
        </div>`;

    const aiId = 'ai-' + Date.now();
    messagesEl.innerHTML += `
        <div class="msg ai" id="${aiId}">
            <div class="msg-avatar">🌾</div>
            <div class="msg-bubble"><span class="typing">Thinking</span></div>
        </div>`;
    messagesEl.scrollTop = messagesEl.scrollHeight;

    const sendBtn = document.getElementById('chatSendBtn');
    sendBtn.disabled = true;

    try {
        const model = document.getElementById('modelSelect').value || 'llama3.2';
        const temp = parseFloat(document.getElementById('temperature').value);
        const useKB = document.getElementById('kbToggle') ? document.getElementById('kbToggle').checked : true;

        const response = await fetch(`${API}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                model,
                temperature: temp,
                history: chatHistory.slice(-10),
                farmer_profile: farmerProfile,
                language: selectedLang,
                use_kb: useKB
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
                    if (data.error) bubble.innerHTML = `<span style="color:var(--red)">❌ ${data.error}</span>`;
                } catch (e) {}
            }
        }

        chatHistory.push({ role: 'user', content: msg });
        chatHistory.push({ role: 'assistant', content: fullResponse });

    } catch (e) {
        document.querySelector(`#${aiId} .msg-bubble`).innerHTML =
            `<span style="color:var(--red)">❌ Failed to connect. Is Ollama running?</span>`;
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
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code style="background:rgba(45,158,107,0.1);padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.85em">$1</code>')
        .replace(/\n/g, '<br>');
}

// ══════════════════════════════════════════
// LANGUAGE
// ══════════════════════════════════════════
function onLangChange() {
    selectedLang = document.getElementById('langSelect').value;
    localStorage.setItem('selectedLang', selectedLang);
}

// ══════════════════════════════════════════
// IMAGE UPLOAD & ANALYSIS
// ══════════════════════════════════════════
function previewImage(input) {
    if (!input.files || !input.files[0]) return;
    pendingImage = input.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        const prev = document.getElementById('imagePreview');
        prev.style.display = 'block';
        prev.innerHTML = `
            <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem 1rem;background:var(--green-lt);border:1px solid var(--green-mid);border-radius:var(--radius-sm);">
                <img src="${e.target.result}" style="width:52px;height:52px;object-fit:cover;border-radius:6px;">
                <div>
                    <div style="color:var(--green);font-weight:600;font-size:0.85rem;">📷 Image ready</div>
                    <div style="color:var(--text-dim);font-size:0.75rem;">${pendingImage.name} — press Send to analyze</div>
                </div>
                <button onclick="clearImagePreview()" style="margin-left:auto;background:none;border:none;color:var(--red);font-size:1.2rem;cursor:pointer;line-height:1;">✕</button>
            </div>`;
    };
    reader.readAsDataURL(pendingImage);
}

function clearImagePreview() {
    pendingImage = null;
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('imagePreview').innerHTML = '';
    document.getElementById('chatImageFile').value = '';
}

async function analyzeImage(question) {
    const messagesEl = document.getElementById('chatMessages');
    const sendBtn = document.getElementById('chatSendBtn');
    sendBtn.disabled = true;

    const reader = new FileReader();
    reader.onload = function(e) {
        messagesEl.innerHTML += `
            <div class="msg user">
                <div class="msg-avatar">👤</div>
                <div class="msg-bubble">
                    <img src="${e.target.result}" style="max-width:180px;border-radius:8px;margin-bottom:0.4rem;display:block;">
                    ${escapeHtml(question)}
                </div>
            </div>`;
    };
    reader.readAsDataURL(pendingImage);

    const aiId = 'ai-' + Date.now();
    messagesEl.innerHTML += `
        <div class="msg ai" id="${aiId}">
            <div class="msg-avatar">🌾</div>
            <div class="msg-bubble"><span class="typing">Analyzing image</span></div>
        </div>`;
    messagesEl.scrollTop = messagesEl.scrollHeight;

    try {
        const formData = new FormData();
        formData.append('file', pendingImage);
        formData.append('question', question);
        formData.append('language', selectedLang);
        formData.append('model', document.getElementById('modelSelect').value || 'llava');

        const resp = await fetch(`${API}/api/analyze-image`, { method: 'POST', body: formData });
        const data = await resp.json();
        document.querySelector(`#${aiId} .msg-bubble`).innerHTML = formatMarkdown(data.analysis);
        chatHistory.push({ role: 'user', content: `[Image] ${question}` });
        chatHistory.push({ role: 'assistant', content: data.analysis });
    } catch (e) {
        document.querySelector(`#${aiId} .msg-bubble`).innerHTML =
            `<span style="color:var(--red)">❌ Image analysis failed. Install llava: <code>ollama pull llava</code></span>`;
    }

    clearImagePreview();
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ══════════════════════════════════════════
// LIVESTOCK — SWITCHING
// ══════════════════════════════════════════
function switchLsTab(idx) {
    document.querySelectorAll('.ls-tab').forEach((t, i) => t.classList.toggle('active', i === idx));
    document.querySelectorAll('.ls-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
}

function switchScanMode(mode, btn) {
    document.getElementById('scanManual').style.display = mode === 'manual' ? 'block' : 'none';
    document.getElementById('scanCsv').style.display = mode === 'csv' ? 'block' : 'none';
    btn.closest('.scan-toggle-bar').querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
}

// ══════════════════════════════════════════
// LIVESTOCK — HEALTH SCAN
// ══════════════════════════════════════════
async function runScan() {
    const fields = ['body_temp','heart_rate','respiratory_rate','activity_level','rumination_min',
        'feed_intake','water_intake','milk_yield','lying_time','steps_count',
        'gait_score','stance_symmetry','stride_length','ambient_temp','humidity_pct'];
    const reading = {};
    fields.forEach(f => { reading[f] = parseFloat(document.getElementById('s_' + f).value); });

    document.getElementById('scanResults').innerHTML =
        '<div class="alert alert-green" style="margin-top:1rem;">🔄 Running biosecurity scan…</div>';

    try {
        const r = await fetch(`${API}/api/livestock/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reading)
        });
        const data = await r.json();
        if (data.error) {
            document.getElementById('scanResults').innerHTML =
                `<div class="alert alert-orange">${data.error}</div>`;
        } else {
            renderScanResults(data);
        }
    } catch (e) {
        document.getElementById('scanResults').innerHTML =
            `<div class="alert alert-red">❌ Scan failed: ${e.message}</div>`;
    }
}

function renderScanResults(data) {
    const h = data.health || {};
    const a = data.anomaly || {};
    const g = data.gait || {};
    const d = data.disease || {};

    const healthClass = h.status_label === 'Healthy' ? 'green' : h.status_label === 'Stressed' ? 'orange' : 'red';
    const anomalyClass = a.is_anomaly ? 'red' : 'green';
    const gaitScore = g.gait_score || 0;
    const gaitClass = gaitScore < 2 ? 'green' : gaitScore < 3 ? 'orange' : 'red';
    const diseaseName = (d.predicted_disease || '').replace(/_/g, ' ') || '—';

    document.getElementById('scanResults').innerHTML = `
        <div class="section-label" style="margin-top:1.5rem;">📊 Biosecurity Scan Results</div>
        <div class="result-grid">
            <div class="r-card ${healthClass}">
                <div class="r-card-title">🏥 Health Status</div>
                <div class="r-card-val">${h.status_label || '—'}</div>
                <div class="r-card-sub">Confidence: ${h.confidence ? (h.confidence*100).toFixed(0) : '?'}%</div>
            </div>
            <div class="r-card ${anomalyClass}">
                <div class="r-card-title">🔍 Anomaly Detection</div>
                <div class="r-card-val">${a.is_anomaly ? '⚠️ Detected' : '✅ Normal'}</div>
                <div class="r-card-sub">Score: ${a.anomaly_score != null ? a.anomaly_score.toFixed(4) : 'N/A'}</div>
            </div>
            <div class="r-card ${gaitClass}">
                <div class="r-card-title">🦶 Gait Analysis</div>
                <div class="r-card-val">${gaitScore.toFixed(1)} / 5</div>
                <div class="r-card-sub">${g.lameness_label || g.gait_label || 'N/A'}</div>
            </div>
            <div class="r-card yellow">
                <div class="r-card-title">🦠 Disease Forecast</div>
                <div class="r-card-val" style="font-size:1.1rem;">${diseaseName}</div>
                <div class="r-card-sub">Risk: ${d.confidence ? (d.confidence*100).toFixed(0) : '?'}%</div>
            </div>
        </div>
        ${data.gait_cv ? `
        <div class="w-card" style="margin-top:0.75rem;">
            <div class="w-card-head">🎥 Computer Vision — Gait</div>
            <p style="color:var(--text-sub);font-size:0.88rem;">
                Locomotion Score: <strong>${data.gait_cv.locomotion_score || data.gait_cv.gait_score || 'N/A'}</strong>
                &nbsp;|&nbsp; ${data.gait_cv.label || ''}
                &nbsp;|&nbsp; ${data.gait_cv.description || ''}
            </p>
        </div>` : ''}
        ${data.behavior ? `
        <div class="w-card" style="margin-top:0.75rem;">
            <div class="w-card-head">🧠 Behavior Analysis</div>
            <p style="color:var(--text-sub);font-size:0.88rem;">
                Pattern: <strong>${(data.behavior.behavior_pattern || '').replace(/_/g,' ')}</strong>
                &nbsp;|&nbsp; Score: <strong>${data.behavior.behavior_health_score || 'N/A'}/100</strong>
                ${data.behavior.alerts && data.behavior.alerts.length ? '<br>Alerts: ' + data.behavior.alerts.map(al => `<span style="color:var(--orange)">${al.type} (${al.severity}): ${al.message}</span>`).join(', ') : ''}
            </p>
        </div>` : ''}`;
}

async function uploadCsv() {
    const file = document.getElementById('csvFile').files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    document.getElementById('scanResults').innerHTML =
        '<div class="alert alert-green" style="margin-top:1rem;">🔄 Processing CSV…</div>';
    try {
        const r = await fetch(`${API}/api/livestock/scan-csv`, { method: 'POST', body: formData });
        const data = await r.json();
        if (data.error) {
            document.getElementById('scanResults').innerHTML = `<div class="alert alert-red">❌ ${data.error}</div>`;
            return;
        }
        let html = `<div class="section-label" style="margin-top:1rem;">📊 Batch Scan Results (${data.total} animals)</div>
            <table class="data-table"><thead><tr>
                <th>#</th><th>Animal</th><th>Health</th><th>Confidence</th><th>Disease Risk</th><th>Gait</th><th>Anomaly</th>
            </tr></thead><tbody>`;
        data.results.forEach(row => {
            html += `<tr><td>${row.id}</td><td>${row.animal_id}</td><td>${row.health}</td><td>${row.confidence}</td><td>${row.disease}</td><td>${row.gait}</td><td>${row.anomaly ? '⚠️' : '✅'}</td></tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('scanResults').innerHTML = html;
    } catch (e) {
        document.getElementById('scanResults').innerHTML =
            '<div class="alert alert-red">❌ CSV upload failed</div>';
    }
}

// ══════════════════════════════════════════
// LIVESTOCK — REFERENCE DATA & RECORDS
// ══════════════════════════════════════════
async function loadLivestockData() {
    const endpoints = [
        ['/api/reference/vaccination', (d) => buildTable(d.schedule, ['disease','vaccine','when','route']), 'vaxScheduleTable'],
        ['/api/reference/diet', (d) => buildTable(d.diet, ['component','qty','examples']), 'dietRefTable'],
        ['/api/reference/mrl', (d) => buildTable(d.guidelines, ['chemical','type','mrl','phi','risk']), 'mrlTable'],
        ['/api/reference/first-aid', (d) => buildTable(d.first_aid, ['emergency','action','time']), 'firstAidTable'],
        ['/api/reference/vet-resources', (d) => buildTable(d.resources, ['service','find','coverage']), 'vetResourcesTable'],
    ];
    for (const [url, fn, el] of endpoints) {
        try {
            const r = await fetch(API + url);
            document.getElementById(el).innerHTML = fn(await r.json());
        } catch (e) {}
    }
    renderVaxRecords();
    renderDietRecords();
    renderFarmInputRecords();
}

function buildTable(rows, keys) {
    const headers = keys.map(k => k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g,' '));
    let html = '<table class="data-table"><thead><tr>' +
        headers.map(h => `<th>${h}</th>`).join('') +
        '</tr></thead><tbody>';
    rows.forEach(row => {
        html += '<tr>' + keys.map(k => `<td>${row[k] || ''}</td>`).join('') + '</tr>';
    });
    return html + '</tbody></table>';
}

function saveVax() {
    const rec = {
        animal: document.getElementById('vaxAnimal').value,
        type: document.getElementById('vaxType').value,
        name: document.getElementById('vaxName').value,
        dose: document.getElementById('vaxDose').value,
        date: document.getElementById('vaxDate').value,
        vet: document.getElementById('vaxVet').value,
        notes: document.getElementById('vaxNotes').value
    };
    if (!rec.name) { alert('Please enter a vaccine/medicine name.'); return; }
    vaxRecords.push(rec);
    localStorage.setItem('vaxRecords', JSON.stringify(vaxRecords));
    renderVaxRecords();
    ['vaxAnimal','vaxName','vaxDose','vaxVet','vaxNotes'].forEach(id => document.getElementById(id).value = '');
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
        notes: document.getElementById('dietNotes').value
    };
    dietRecords.push(rec);
    localStorage.setItem('dietRecords', JSON.stringify(dietRecords));
    renderDietRecords();
    ['dietAnimal','dietNotes'].forEach(id => document.getElementById(id).value = '');
}

function saveFarmInput() {
    const rec = {
        type: document.getElementById('fiType').value,
        name: document.getElementById('fiName').value,
        dose: document.getElementById('fiDose').value,
        field: document.getElementById('fiField').value,
        date: document.getElementById('fiDate').value,
        phi: document.getElementById('fiPhi').value,
        notes: document.getElementById('fiNotes').value
    };
    if (!rec.name) { alert('Please enter a product name.'); return; }
    farmInputRecords.push(rec);
    localStorage.setItem('farmInputRecords', JSON.stringify(farmInputRecords));
    renderFarmInputRecords();
    ['fiName','fiDose','fiField','fiNotes'].forEach(id => document.getElementById(id).value = '');
}

function renderVaxRecords() {
    if (!vaxRecords.length) { document.getElementById('vaxRecords').innerHTML = ''; return; }
    document.getElementById('vaxRecords').innerHTML =
        `<div class="section-label">📋 Saved Records (${vaxRecords.length})</div>` +
        buildTable(vaxRecords, ['date','animal','type','name','dose','vet']);
}
function renderDietRecords() {
    if (!dietRecords.length) { document.getElementById('dietRecords').innerHTML = ''; return; }
    document.getElementById('dietRecords').innerHTML =
        `<div class="section-label">📋 Diet Logs (${dietRecords.length})</div>` +
        buildTable(dietRecords, ['date','animal','lactation','green','dry','concentrates','water']);
}
function renderFarmInputRecords() {
    if (!farmInputRecords.length) { document.getElementById('farmInputRecords').innerHTML = ''; return; }
    document.getElementById('farmInputRecords').innerHTML =
        `<div class="section-label">📋 Application Records (${farmInputRecords.length})</div>` +
        buildTable(farmInputRecords, ['date','type','name','dose','field','phi']);
}

// ── Emergency Vet ──
function findVet() {
    const location = document.getElementById('vetLocation').value.trim();
    if (!location) { alert('Please enter your location'); return; }
    const url = `https://www.google.com/maps/search/veterinary+hospital+near+${encodeURIComponent(location)}`;
    document.getElementById('vetResults').innerHTML = `
        <div class="w-card" style="margin-top:1rem;">
            <p style="color:var(--text-sub);font-size:0.9rem;">Searching for <strong>veterinary services</strong> near <strong>${escapeHtml(location)}</strong></p>
            <a href="${url}" target="_blank" class="btn-primary" style="margin-top:0.8rem;display:inline-flex;text-decoration:none;">
                🗺️ Open in Google Maps →
            </a>
        </div>`;
}
// ══════════════════════════════════════════
// WEATHER MINI PANEL
// ══════════════════════════════════════════
async function loadWeatherMini(region) {
    try {
        const r = await fetch(`${API}/api/weather/${region}`);
        const data = await r.json();

        if (!data.forecast || !data.forecast.length) return;

        const today = data.forecast[0];

        const tempEl = document.getElementById("weatherMiniTemp");
        const descEl = document.getElementById("weatherMiniDesc");
        const humEl = document.getElementById("weatherMiniHumidity");
        const windEl = document.getElementById("weatherMiniWind");
        const adviceEl = document.getElementById("weatherMiniAdvice");

        if (!tempEl) return; // safety check if UI not loaded

        tempEl.innerText = today.temperature + "°C";
        descEl.innerText = today.weather;
        humEl.innerText = today.humidity + "%";
        windEl.innerText = today.wind_speed + " m/s";

        let advice = "Conditions normal for farming.";

        if (today.temperature > 38)
            advice = "⚠ High heat — Irrigate early morning.";

        if (today.wind_speed > 6)
            advice = "⚠ Strong winds — Avoid pesticide spray.";

        if (today.humidity > 80)
            advice = "⚠ High humidity — Fungal disease risk.";

        adviceEl.innerText = advice;

    } catch (err) {
        console.log("Weather load error:", err);
    }
}