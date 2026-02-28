print("🚀 RUNNING FILE:", __file__)
import json, pickle, requests, numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    from livestock_biosecurity.models import load_livestock_models, GaitAnalyzer, BehaviorAnalyzer
    LIVESTOCK_AVAILABLE = True
except:
    try:
        from livestock_app import load_livestock_models, GaitAnalyzer, BehaviorAnalyzer
        LIVESTOCK_AVAILABLE = True
    except:
        LIVESTOCK_AVAILABLE = False

OLLAMA_URL = "http://localhost:11434"
VECTOR_STORE_DIR = "agricultural_vector_store"
SYSTEM_PROMPT = "You are KrishiSakhi, an advanced AI agricultural assistant. Provide practical farming advice warmly and clearly."

class VectorStoreManager:
    def __init__(self, d):
        self.vector_store_dir = Path(d)
        self.index = None
        self.documents = []
        self.chunks_metadata = []
        self.loaded = False
    def load_vector_store(self):
        try:
            import faiss
            ip = self.vector_store_dir / "faiss_index.bin"
            mp = self.vector_store_dir / "metadata.pkl"
            if not ip.exists() or not mp.exists(): return False
            self.index = faiss.read_index(str(ip))
            with open(mp, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.chunks_metadata = data['chunks_metadata']
            self.loaded = True
            print(f"Vector store loaded: {len(self.documents)} docs")
            return True
        except Exception as e:
            print(f"Vector store error: {e}")
            return False
    def get_embedding(self, text):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": "nomic-embed-text", "prompt": text}, timeout=30)
            r.raise_for_status()
            return r.json()["embedding"]
        except: return None
    def search(self, query, k=3):
        if not self.loaded: return []
        import faiss
        emb = self.get_embedding(query)
        if emb is None: return []
        qa = np.array([emb]).astype('float32')
        faiss.normalize_L2(qa)
        scores, indices = self.index.search(qa, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and score > 0.1:
                results.append({"content": self.documents[idx], "metadata": self.chunks_metadata[idx], "score": float(score)})
        return results

def get_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", []) if 'embed' not in m["name"].lower()]
        return []
    except: return []

def check_ollama():
    try: return requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).status_code == 200
    except: return False

def build_prompt(query, docs):
    ctx = ""
    if docs:
        ctx = "\nRelevant Knowledge:\n"
        for i, d in enumerate(docs, 1):
            fn = d['metadata'].get('filename', 'Unknown')
            preview = d['content'][:400]
            ctx += f"\n[Source {i} - {fn}]:\n{preview}\n"
    return f"{ctx}\n\nFarmer Question: {query}"

def stream_ollama(model, messages, temp=0.7):
    payload = {"model": model, "messages": messages, "stream": True, "options": {"temperature": temp, "system": SYSTEM_PROMPT}}
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]
                        if chunk.get("done"): break
                    except: continue
    except Exception as e: yield f"Error: {str(e)}"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

vs = VectorStoreManager(VECTOR_STORE_DIR)
vs.load_vector_store()
lm = None
if LIVESTOCK_AVAILABLE:
    try:
        lm = load_livestock_models('trained_models')
        print("Livestock models loaded")
    except Exception as e: print(f"Livestock models error: {e}")

class Msg(BaseModel):
    role: str
    content: str
    base64_image: Optional[str] = None

class ChatReq(BaseModel):
    question: str
    chat_history: List[Msg] = []
    model: str = "llama3.2"
    temperature: float = 0.7
    use_knowledge_base: bool = True
    k_documents: int = 3
    base64_image: Optional[str] = None

class LivestockReq(BaseModel):
    body_temp: float; heart_rate: float; respiratory_rate: float
    activity_level: float; rumination_min: float; feed_intake: float
    water_intake: float; milk_yield: float; lying_time: float
    steps_count: float; gait_score: float; stance_symmetry: float
    stride_length: float; ambient_temp: float; humidity_pct: float; thi_index: float

@app.get("/api/status")
def status():
    return {"ollama": check_ollama(), "vector_store": vs.loaded, "livestock": lm is not None, "models": get_models()}

@app.post("/api/chat")
def chat(req: ChatReq):
    docs = vs.search(req.question, req.k_documents) if req.use_knowledge_base and vs.loaded else []
    prompt = build_prompt(req.question, docs)
    msgs = []
    for m in req.chat_history:
        msg = {"role": m.role, "content": m.content}
        if m.base64_image: msg["images"] = [m.base64_image]
        msgs.append(msg)
    cur = {"role": "user", "content": prompt}
    if req.base64_image: cur["images"] = [req.base64_image]
    msgs.append(cur)
    return StreamingResponse(stream_ollama(req.model, msgs, req.temperature), media_type="text/plain")

@app.post("/api/livestock/scan")
def scan(req: LivestockReq):
    if not lm: return {"error": "Livestock models not available"}
    r = req.dict()
    return {"health": lm['health_predictor'].predict(r), "anomaly": lm['anomaly_detector'].predict(r),
            "gait": lm['gait_predictor'].predict(r), "disease": lm['disease_forecaster'].predict(r),
            "gait_cv": GaitAnalyzer().analyze_gait(r), "behavior": BehaviorAnalyzer().analyze_behavior(r)}

#app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
