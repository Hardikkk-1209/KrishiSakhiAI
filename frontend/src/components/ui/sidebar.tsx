import { useState } from "react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import {
    MessageSquare,
    Beef,
    ChevronDown,
    Thermometer,
    Globe,
    Bot,
    SlidersHorizontal,
    LogOut,
    Scan,
    Syringe,
    Leaf,
    Tractor,
    AlertTriangle,
    User,
} from "lucide-react";

interface SidebarProps {
    activeMode: 'chat' | 'livestock';
    onModeChange: (mode: 'chat' | 'livestock') => void;
    model: string;
    onModelChange: (m: string) => void;
    models: string[];
    language: string;
    onLanguageChange: (l: string) => void;
    temperature: number;
    onTemperatureChange: (t: number) => void;
    farmerProfile: Record<string, string> | null;
    onResetProfile: () => void;
    activeLsTab: number;
    onLsTabChange: (i: number) => void;
}

export function Sidebar({
    activeMode, onModeChange,
    model, onModelChange, models,
    language, onLanguageChange,
    temperature, onTemperatureChange,
    farmerProfile, onResetProfile,
    activeLsTab, onLsTabChange,
}: SidebarProps) {
    const [isSettingsOpen, setIsSettingsOpen] = useState(true);

    const livestockTabs = [
        { icon: <Scan size={16} />, label: "Health Scanner" },
        { icon: <Syringe size={16} />, label: "Vaccination" },
        { icon: <Leaf size={16} />, label: "Feed & Diet" },
        { icon: <Tractor size={16} />, label: "Farm Inputs" },
        { icon: <AlertTriangle size={16} />, label: "Emergency Vet" },
    ];

    return (
        <div
            style={{
                width: 280,
                minWidth: 280,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                background: '#111113',
                borderRight: '1px solid rgba(255,255,255,0.06)',
                overflow: 'hidden',
            }}
        >
            {/* Brand */}
            <div style={{ padding: '20px 20px 16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div
                        style={{
                            width: 40, height: 40, borderRadius: 12,
                            background: 'linear-gradient(135deg, #22c55e, #16a34a)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: 20, boxShadow: '0 4px 12px rgba(34,197,94,0.25)',
                        }}
                    >
                        🌾
                    </div>
                    <div>
                        <div style={{ fontSize: 15, fontWeight: 700, color: '#fff', letterSpacing: '-0.02em' }}>
                            KrishiSakhiAI
                        </div>
                        <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', fontWeight: 500 }}>
                            Smart Farming Assistant
                        </div>
                    </div>
                </div>
            </div>

            {/* Farmer Badge */}
            {farmerProfile && (
                <div
                    style={{
                        margin: '0 16px 12px',
                        padding: '10px 14px',
                        borderRadius: 12,
                        background: 'rgba(34,197,94,0.08)',
                        border: '1px solid rgba(34,197,94,0.15)',
                    }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <User size={16} color="#4ade80" />
                        <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 13, fontWeight: 600, color: '#4ade80' }}>
                                {farmerProfile.name}
                            </div>
                            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)' }}>
                                📍 {farmerProfile.region} • {farmerProfile.land_size} acres
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Mode Switch */}
            <div style={{ padding: '0 16px', marginBottom: 12 }}>
                <div
                    style={{
                        display: 'flex', gap: 4, padding: 4, borderRadius: 12,
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.06)',
                    }}
                >
                    {(['chat', 'livestock'] as const).map((mode) => (
                        <button
                            key={mode}
                            onClick={() => onModeChange(mode)}
                            style={{
                                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                                padding: '10px 8px', borderRadius: 10, border: 'none', cursor: 'pointer',
                                fontSize: 13, fontWeight: 600, fontFamily: 'inherit',
                                background: activeMode === mode ? '#22c55e' : 'transparent',
                                color: activeMode === mode ? '#fff' : 'rgba(255,255,255,0.45)',
                                boxShadow: activeMode === mode ? '0 2px 8px rgba(34,197,94,0.3)' : 'none',
                                transition: 'all 0.2s',
                            }}
                        >
                            {mode === 'chat' ? <><MessageSquare size={15} /> AI Chat</> : <><Beef size={15} /> Livestock</>}
                        </button>
                    ))}
                </div>
            </div>

            {/* Livestock Tabs */}
            <AnimatePresence>
                {activeMode === 'livestock' && (
                    <motion.div
                        style={{ padding: '0 12px', marginBottom: 8 }}
                        initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                    >
                        {livestockTabs.map((tab, i) => (
                            <button
                                key={i}
                                onClick={() => onLsTabChange(i)}
                                style={{
                                    width: '100%', display: 'flex', alignItems: 'center', gap: 10,
                                    padding: '10px 14px', borderRadius: 10, border: 'none', cursor: 'pointer',
                                    fontSize: 13, fontWeight: 500, fontFamily: 'inherit', marginBottom: 2,
                                    background: activeLsTab === i ? 'rgba(255,255,255,0.08)' : 'transparent',
                                    color: activeLsTab === i ? '#fff' : 'rgba(255,255,255,0.4)',
                                    transition: 'all 0.2s',
                                }}
                                onMouseEnter={(e) => { if (activeLsTab !== i) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
                                onMouseLeave={(e) => { if (activeLsTab !== i) e.currentTarget.style.background = 'transparent'; }}
                            >
                                {tab.icon}
                                {tab.label}
                            </button>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Spacer */}
            <div style={{ flex: 1 }} />

            {/* Settings */}
            <div style={{ padding: '0 12px 16px' }}>
                <button
                    onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                    style={{
                        width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '10px 14px', borderRadius: 10, border: 'none', cursor: 'pointer',
                        fontSize: 13, fontWeight: 500, fontFamily: 'inherit',
                        background: 'transparent', color: 'rgba(255,255,255,0.45)',
                        transition: 'all 0.2s',
                    }}
                >
                    <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <SlidersHorizontal size={15} /> Settings
                    </span>
                    <ChevronDown size={15} style={{ transform: isSettingsOpen ? 'rotate(180deg)' : 'none', transition: '0.2s' }} />
                </button>

                <AnimatePresence>
                    {isSettingsOpen && (
                        <motion.div
                            style={{ padding: '8px 6px 0', display: 'flex', flexDirection: 'column', gap: 14 }}
                            initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                        >
                            {/* Model Select */}
                            <div>
                                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'rgba(255,255,255,0.35)', fontWeight: 600, marginBottom: 6, paddingLeft: 2 }}>
                                    <Bot size={12} /> AI Model
                                </label>
                                <select
                                    value={model} onChange={(e) => onModelChange(e.target.value)}
                                    style={{
                                        width: '100%', padding: '8px 10px', borderRadius: 8,
                                        background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
                                        color: 'rgba(255,255,255,0.8)', fontSize: 13, fontFamily: 'inherit',
                                        outline: 'none', cursor: 'pointer',
                                    }}
                                >
                                    {models.map(m => <option key={m} value={m} style={{ background: '#111', color: '#fff' }}>{m}</option>)}
                                </select>
                            </div>

                            {/* Language Select */}
                            <div>
                                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'rgba(255,255,255,0.35)', fontWeight: 600, marginBottom: 6, paddingLeft: 2 }}>
                                    <Globe size={12} /> Language / भाषा
                                </label>
                                <select
                                    value={language} onChange={(e) => onLanguageChange(e.target.value)}
                                    style={{
                                        width: '100%', padding: '8px 10px', borderRadius: 8,
                                        background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
                                        color: 'rgba(255,255,255,0.8)', fontSize: 13, fontFamily: 'inherit',
                                        outline: 'none', cursor: 'pointer',
                                    }}
                                >
                                    <option value="English" style={{ background: '#111', color: '#fff' }}>🇬🇧 English</option>
                                    <option value="Hindi" style={{ background: '#111', color: '#fff' }}>🇮🇳 हिंदी (Hindi)</option>
                                    <option value="Marathi" style={{ background: '#111', color: '#fff' }}>🏳️ मराठी (Marathi)</option>
                                </select>
                            </div>

                            {/* Temperature */}
                            <div>
                                <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11, color: 'rgba(255,255,255,0.35)', fontWeight: 600, marginBottom: 6, paddingLeft: 2 }}>
                                    <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><Thermometer size={12} /> Temperature</span>
                                    <span style={{ color: '#4ade80', fontWeight: 700 }}>{temperature.toFixed(2)}</span>
                                </label>
                                <input
                                    type="range" min="0" max="1" step="0.05"
                                    value={temperature} onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
                                    style={{ width: '100%', accentColor: '#22c55e', height: 4 }}
                                />
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Reset */}
                {farmerProfile && (
                    <button
                        onClick={onResetProfile}
                        style={{
                            width: '100%', display: 'flex', alignItems: 'center', gap: 8,
                            padding: '10px 14px', borderRadius: 10, border: 'none', cursor: 'pointer',
                            fontSize: 13, fontWeight: 500, fontFamily: 'inherit', marginTop: 8,
                            background: 'transparent', color: 'rgba(239,68,68,0.5)',
                            transition: 'all 0.2s',
                        }}
                    >
                        <LogOut size={15} /> Reset Profile
                    </button>
                )}
            </div>
        </div>
    );
}
