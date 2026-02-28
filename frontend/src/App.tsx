import { useState, useEffect } from 'react';
import { Sidebar } from '@/components/ui/sidebar';
import { AnimatedAIChat } from '@/components/ui/animated-ai-chat';
import { LivestockHub } from '@/components/ui/livestock-hub';

function App() {
  const [activeMode, setActiveMode] = useState<'chat' | 'livestock'>('chat');
  const [model, setModel] = useState('llama3.2:1b');
  const [models, setModels] = useState<string[]>(['llama3.2:1b']);
  const [language, setLanguage] = useState(
    () => localStorage.getItem('selectedLang') || 'English'
  );
  const [temperature, setTemperature] = useState(0.7);
  const [farmerProfile, setFarmerProfile] = useState<Record<string, string> | null>(
    () => {
      const saved = localStorage.getItem('farmerProfile');
      return saved ? JSON.parse(saved) : null;
    }
  );
  const [activeLsTab, setActiveLsTab] = useState(0);
  const [ollamaStatus, setOllamaStatus] = useState<'connected' | 'disconnected'>('disconnected');

  // Check health + fetch models on mount
  useEffect(() => {
    fetch('/api/health')
      .then(r => r.json())
      .then(d => setOllamaStatus(d.ollama ? 'connected' : 'disconnected'))
      .catch(() => setOllamaStatus('disconnected'));

    fetch('/api/models')
      .then(r => r.json())
      .then(d => {
        if (d.models?.length) {
          setModels(d.models);
          setModel(d.models[0]);
        }
      })
      .catch(() => { });
  }, []);

  // Persist language
  useEffect(() => {
    localStorage.setItem('selectedLang', language);
  }, [language]);

  const resetProfile = () => {
    localStorage.removeItem('farmerProfile');
    setFarmerProfile(null);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0a0a0b]">
      {/* Sidebar */}
      <Sidebar
        activeMode={activeMode}
        onModeChange={setActiveMode}
        model={model}
        onModelChange={setModel}
        models={models}
        language={language}
        onLanguageChange={setLanguage}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        farmerProfile={farmerProfile}
        onResetProfile={resetProfile}
        activeLsTab={activeLsTab}
        onLsTabChange={setActiveLsTab}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeMode === 'chat' ? (
          <AnimatedAIChat
            language={language}
            model={model}
            temperature={temperature}
            farmerProfile={farmerProfile}
          />
        ) : (
          <LivestockHub activeTab={activeLsTab} language={language} />
        )}
      </div>
    </div>
  );
}

export default App;
