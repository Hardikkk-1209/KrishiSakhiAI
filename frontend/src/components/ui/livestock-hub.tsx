import { useState } from "react";
import { motion } from "framer-motion";

interface LivestockHubProps {
    activeTab: number;
    language: string;
}

const scanFields = [
    { id: 's_body_temp', label: '🌡️ Body Temp (°C)', val: 38.8, step: 0.1 },
    { id: 's_heart_rate', label: '❤️ Heart Rate (bpm)', val: 60, step: 1 },
    { id: 's_respiratory_rate', label: '🫁 Resp. Rate (/min)', val: 22, step: 1 },
    { id: 's_activity_level', label: '⚡ Activity (%)', val: 68, step: 1 },
    { id: 's_rumination_min', label: '🔄 Rumination (min)', val: 450, step: 10 },
    { id: 's_feed_intake', label: '🍽️ Feed Intake (kg)', val: 22, step: 0.5 },
    { id: 's_water_intake', label: '💧 Water Intake (L)', val: 55, step: 1 },
    { id: 's_milk_yield', label: '🥛 Milk Yield (L)', val: 20, step: 0.5 },
    { id: 's_lying_time', label: '🛌 Lying Time (hrs)', val: 12, step: 0.5 },
    { id: 's_steps_count', label: '👣 Steps Count', val: 3500, step: 100 },
    { id: 's_gait_score', label: '🦶 Gait Score', val: 1.2, step: 0.1 },
    { id: 's_stance_symmetry', label: '⚖️ Stance Symmetry', val: 0.95, step: 0.01 },
    { id: 's_stride_length', label: '📏 Stride Length (m)', val: 1.5, step: 0.1 },
    { id: 's_ambient_temp', label: '🌡️ Ambient Temp (°C)', val: 30, step: 0.5 },
    { id: 's_humidity_pct', label: '💨 Humidity (%)', val: 65, step: 1 },
];

const inputStyle: React.CSSProperties = {
    width: '100%', padding: '10px 12px', borderRadius: 8,
    background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
    color: 'rgba(255,255,255,0.85)', fontSize: 14, fontFamily: 'inherit',
    outline: 'none',
};

const labelStyle: React.CSSProperties = {
    fontSize: 12, color: 'rgba(255,255,255,0.4)', fontWeight: 600, marginBottom: 6, display: 'block',
};

const cardStyle: React.CSSProperties = {
    background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 16, padding: 24, marginBottom: 16,
};

export function LivestockHub({ activeTab, language }: LivestockHubProps) {
    const [scanValues, setScanValues] = useState<Record<string, number>>(
        Object.fromEntries(scanFields.map(f => [f.id, f.val]))
    );
    const [scanResult, setScanResult] = useState<string | null>(null);
    const [isScanning, setIsScanning] = useState(false);

    const runScan = async () => {
        setIsScanning(true);
        try {
            const resp = await fetch('/api/livestock/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    body_temp: scanValues.s_body_temp,
                    heart_rate: scanValues.s_heart_rate,
                    respiratory_rate: scanValues.s_respiratory_rate,
                    activity_level: scanValues.s_activity_level,
                    rumination_min: scanValues.s_rumination_min,
                    feed_intake: scanValues.s_feed_intake,
                    water_intake: scanValues.s_water_intake,
                    milk_yield: scanValues.s_milk_yield,
                    lying_time: scanValues.s_lying_time,
                    steps_count: scanValues.s_steps_count,
                    gait_score: scanValues.s_gait_score,
                    stance_symmetry: scanValues.s_stance_symmetry,
                    stride_length: scanValues.s_stride_length,
                    ambient_temp: scanValues.s_ambient_temp,
                    humidity_pct: scanValues.s_humidity_pct,
                }),
            });
            const data = await resp.json();
            setScanResult(JSON.stringify(data, null, 2));
        } catch {
            setScanResult('❌ Failed to connect to server.');
        }
        setIsScanning(false);
    };

    return (
        <div style={{ flex: 1, overflowY: 'auto', padding: 32, color: '#fafafa' }}>
            <div style={{ maxWidth: 1000, margin: '0 auto' }}>
                {/* Header */}
                <motion.div
                    style={{ marginBottom: 28, textAlign: 'center' }}
                    initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                >
                    <h1 style={{ fontSize: 28, fontWeight: 800, background: 'linear-gradient(135deg, #6ee7b7, #22c55e)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        🐄 Livestock Biosecurity Hub
                    </h1>
                    <p style={{ fontSize: 13, color: 'rgba(255,255,255,0.35)', marginTop: 6 }}>
                        AI diagnostics • Vaccination • Diet • Farm inputs • Emergency vet
                    </p>
                </motion.div>

                {/* Tab 0: Health Scanner */}
                {activeTab === 0 && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 20 }}>
                            {scanFields.map((f) => (
                                <div key={f.id}>
                                    <label style={labelStyle}>{f.label}</label>
                                    <input
                                        type="number"
                                        value={scanValues[f.id]}
                                        step={f.step}
                                        onChange={(e) => setScanValues(v => ({ ...v, [f.id]: parseFloat(e.target.value) || 0 }))}
                                        style={inputStyle}
                                    />
                                </div>
                            ))}
                        </div>
                        <button
                            onClick={runScan}
                            disabled={isScanning}
                            style={{
                                width: '100%', padding: 14, borderRadius: 12, border: 'none', cursor: 'pointer',
                                background: '#22c55e', color: '#fff', fontSize: 15, fontWeight: 700, fontFamily: 'inherit',
                                boxShadow: '0 4px 16px rgba(34,197,94,0.3)',
                                opacity: isScanning ? 0.6 : 1,
                            }}
                        >
                            {isScanning ? '🔄 Scanning...' : '🔬 Run Biosecurity Scan'}
                        </button>
                        {scanResult && (
                            <motion.pre
                                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                                style={{
                                    marginTop: 20, padding: 20, borderRadius: 12,
                                    background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
                                    fontSize: 13, color: 'rgba(255,255,255,0.65)', overflowX: 'auto', whiteSpace: 'pre-wrap',
                                }}
                            >
                                {scanResult}
                            </motion.pre>
                        )}
                    </motion.div>
                )}

                {/* Tab 1: Vaccination */}
                {activeTab === 1 && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <div style={cardStyle}>
                            <h3 style={{ fontSize: 16, fontWeight: 700, color: '#4ade80', marginBottom: 20 }}>💉 Log Vaccination / Medicine</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
                                <div><label style={labelStyle}>🐄 Animal ID</label><input style={inputStyle} placeholder="COW-001" /></div>
                                <div>
                                    <label style={labelStyle}>Type</label>
                                    <select style={inputStyle}>
                                        <option>Vaccination</option><option>Deworming</option><option>Medicine</option><option>Supplement</option>
                                    </select>
                                </div>
                                <div><label style={labelStyle}>Vaccine / Medicine</label><input style={inputStyle} placeholder="FMD Vaccine" /></div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
                                <div><label style={labelStyle}>Dosage</label><input style={inputStyle} placeholder="5ml" /></div>
                                <div><label style={labelStyle}>Date</label><input type="date" style={inputStyle} /></div>
                                <div><label style={labelStyle}>Vet Name</label><input style={inputStyle} placeholder="Dr. Patil" /></div>
                            </div>
                            <div style={{ marginBottom: 16 }}>
                                <label style={labelStyle}>📝 Notes</label>
                                <textarea style={{ ...inputStyle, minHeight: 70, resize: 'vertical' as const }} placeholder="Any special notes..." />
                            </div>
                            <button style={{ width: '100%', padding: 12, borderRadius: 10, border: 'none', cursor: 'pointer', background: '#22c55e', color: '#fff', fontSize: 14, fontWeight: 600, fontFamily: 'inherit' }}>
                                💾 Save Record
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Tab 2: Feed & Diet */}
                {activeTab === 2 && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <div style={cardStyle}>
                            <h3 style={{ fontSize: 16, fontWeight: 700, color: '#4ade80', marginBottom: 20 }}>🥬 Log Daily Diet</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
                                <div><label style={labelStyle}>🐄 Animal ID</label><input style={inputStyle} placeholder="COW-001" /></div>
                                <div><label style={labelStyle}>Date</label><input type="date" style={inputStyle} /></div>
                                <div>
                                    <label style={labelStyle}>Lactation Stage</label>
                                    <select style={inputStyle}>
                                        <option>Early (0-100 days)</option><option>Mid (100-200)</option><option>Late (200-305)</option><option>Dry Period</option>
                                    </select>
                                </div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
                                <div><label style={labelStyle}>🌿 Green Fodder (kg)</label><input type="number" defaultValue={30} style={inputStyle} /></div>
                                <div><label style={labelStyle}>🌾 Dry Fodder (kg)</label><input type="number" defaultValue={6} style={inputStyle} /></div>
                                <div><label style={labelStyle}>🌽 Concentrates (kg)</label><input type="number" defaultValue={3} style={inputStyle} /></div>
                                <div><label style={labelStyle}>💧 Water (L)</label><input type="number" defaultValue={60} style={inputStyle} /></div>
                            </div>
                            <button style={{ width: '100%', padding: 12, borderRadius: 10, border: 'none', cursor: 'pointer', background: '#22c55e', color: '#fff', fontSize: 14, fontWeight: 600, fontFamily: 'inherit' }}>
                                💾 Save Diet Log
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Tab 3: Farm Inputs */}
                {activeTab === 3 && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <div style={cardStyle}>
                            <h3 style={{ fontSize: 16, fontWeight: 700, color: '#4ade80', marginBottom: 20 }}>🚜 Log Pesticide / Fertilizer</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
                                <div>
                                    <label style={labelStyle}>Type</label>
                                    <select style={inputStyle}>
                                        <option>Pesticide</option><option>Fertilizer</option><option>Herbicide</option><option>Bio-pesticide</option>
                                    </select>
                                </div>
                                <div><label style={labelStyle}>Product Name</label><input style={inputStyle} placeholder="Imidacloprid 17.8SL" /></div>
                                <div><label style={labelStyle}>Dosage</label><input style={inputStyle} placeholder="0.5ml/L" /></div>
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
                                <div><label style={labelStyle}>Field / Area</label><input style={inputStyle} placeholder="Plot A" /></div>
                                <div><label style={labelStyle}>Date</label><input type="date" style={inputStyle} /></div>
                                <div><label style={labelStyle}>PHI (days)</label><input type="number" defaultValue={14} style={inputStyle} /></div>
                            </div>
                            <button style={{ width: '100%', padding: 12, borderRadius: 10, border: 'none', cursor: 'pointer', background: '#22c55e', color: '#fff', fontSize: 14, fontWeight: 600, fontFamily: 'inherit' }}>
                                💾 Save Record
                            </button>
                        </div>
                    </motion.div>
                )}

                {/* Tab 4: Emergency Vet */}
                {activeTab === 4 && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                        <div
                            style={{
                                padding: 28, borderRadius: 16, textAlign: 'center', marginBottom: 20,
                                background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.15)',
                            }}
                        >
                            <div style={{ fontSize: 40, marginBottom: 8 }}>🚨</div>
                            <h3 style={{ fontSize: 20, fontWeight: 800, color: '#f87171' }}>Emergency Helplines</h3>
                            <p style={{ color: 'rgba(255,255,255,0.65)', fontSize: 15, marginTop: 10 }}>
                                📞 <strong>1962</strong> — National Animal Helpline
                            </p>
                            <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 13, marginTop: 4 }}>
                                📞 1800-121-3456 — NDDB (Dairy)
                            </p>
                        </div>
                        <div style={cardStyle}>
                            <h3 style={{ fontSize: 16, fontWeight: 700, color: '#4ade80', marginBottom: 16 }}>🗺️ Find Nearest Vet</h3>
                            <div style={{ display: 'flex', gap: 12 }}>
                                <input style={{ ...inputStyle, flex: 1 }} placeholder="Your location / village / pincode..." />
                                <button style={{ padding: '10px 20px', borderRadius: 10, border: 'none', cursor: 'pointer', background: '#22c55e', color: '#fff', fontSize: 14, fontWeight: 600, fontFamily: 'inherit', whiteSpace: 'nowrap' as const }}>
                                    🔍 Search
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>
        </div>
    );
}
