import json, pickle, base64, requests, numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd

# ── Livestock models ─────────────────────────────────────────────
import joblib
from pathlib import Path

LIVESTOCK_AVAILABLE = True

def load_livestock_models(path):
    path = Path(path)

    return {
        "health_predictor": joblib.load(path / "health_predictor.pkl"),
        "anomaly_detector": joblib.load(path / "anomaly_detector.pkl"),
        "disease_forecaster": joblib.load(path / "disease_forecaster.pkl"),
        "gait_predictor": joblib.load(path / "gait_predictor.pkl"),
    }

class GaitAnalyzer:
    def analyze_gait(self, data):
        return {
            "gait_score": data.get("gait_score", 1.0),
            "risk": "Low" if data.get("gait_score", 1) <= 2 else "High"
        }

class BehaviorAnalyzer:
    def analyze_behavior(self, data):
        return {
            "activity": data.get("activity_level", 70),
            "rumination": data.get("rumination_min", 450),
            "status": "Normal"
        }

OLLAMA_URL = "http://localhost:11434"
VECTOR_STORE_DIR = "agricultural_vector_store"
SYSTEM_PROMPT = """You are KrishiSakhi, an advanced AI agricultural assistant with deep expertise in Indian farming.
You provide warm, practical advice on crop management, pest/disease identification, weather-based recommendations,
sustainable methods, cost-effective solutions, soil health, irrigation, and market trends.
Always prioritize farmer safety, environmental sustainability, and local conditions.
Respond in the language requested by the user."""

# ── Maharashtra Regions ──────────────────────────────────────────
REGIONS = {
    "Vidarbha": {"climate": "Semi-arid, hot summers", "soil": "Black cotton soil (Vertisol)",
        "common": ["Cotton", "Soybean", "Oranges", "Tur Dal", "Wheat", "Sorghum", "Sunflower", "Chickpea"],
        "rare": ["Stevia", "Safflower", "Isabgol", "Quinoa"]},
    "Marathwada": {"climate": "Dry, drought-prone", "soil": "Medium-deep black soil",
        "common": ["Soybean", "Cotton", "Sorghum", "Tur Dal", "Wheat", "Onion", "Sweet Lime"],
        "rare": ["Ashwagandha", "Safed Musli", "Aloe Vera", "Moringa"]},
    "Western Maharashtra": {"climate": "Moderate, good rainfall", "soil": "Red laterite + black mix",
        "common": ["Sugarcane", "Onion", "Grapes", "Pomegranate", "Tomato", "Wheat", "Jowar"],
        "rare": ["Dragon Fruit", "Strawberry", "Blueberry", "Lavender"]},
    "Konkan": {"climate": "Humid tropical, heavy monsoon", "soil": "Laterite, coastal alluvial",
        "common": ["Rice", "Alphonso Mango", "Coconut", "Cashew", "Jackfruit", "Kokum", "Betel nut"],
        "rare": ["Vanilla", "Black Pepper", "Nutmeg", "Cardamom"]},
    "Nashik": {"climate": "Mild, Mediterranean-like", "soil": "Sandy loam + black soil",
        "common": ["Grapes", "Onion", "Tomato", "Sweet Corn", "Wheat", "Pomegranate", "Maize"],
        "rare": ["Hops", "Asparagus", "Broccoli", "Artichoke"]},
    "Pune": {"climate": "Semi-arid to moderate", "soil": "Mixed red and black",
        "common": ["Sugarcane", "Onion", "Grapes", "Wheat", "Tomato", "Ginger", "Turmeric"],
        "rare": ["Saffron", "Chia Seeds", "Arrowroot", "Taro"]}
}

RARE_CROP_DISEASES = {
    "Stevia": {"diseases": [
        {"name": "Septoria Leaf Spot", "symptoms": "Brown spots with yellow halos on older leaves",
         "prevention": "Avoid overhead irrigation, plant spacing 45cm, crop rotation",
         "treatment": "Remove infected leaves, improve air circulation",
         "pesticides": "Mancozeb 75WP @ 2g/L, Copper oxychloride @ 3g/L"},
        {"name": "Root Rot (Phytophthora)", "symptoms": "Wilting, yellowing, dark brown roots",
         "prevention": "Well-drained soil, raised beds, avoid waterlogging",
         "treatment": "Drench with Metalaxyl @ 1g/L, remove severely affected plants",
         "pesticides": "Ridomil Gold @ 2g/L soil drench"}]},
    "Dragon Fruit": {"diseases": [
        {"name": "Stem Canker (Gleosporium)", "symptoms": "Orange-yellow spots becoming sunken lesions on stems",
         "prevention": "Avoid stem injuries, proper staking, prune old stems",
         "treatment": "Scrape and paint lesions with Bordeaux paste",
         "pesticides": "Propiconazole 25EC @ 1mL/L, Carbendazim 50WP @ 1g/L"},
        {"name": "Anthracnose", "symptoms": "Dark water-soaked lesions on fruit, stem rot",
         "prevention": "Avoid excess humidity, proper drainage, sanitation",
         "treatment": "Mancozeb + Carbendazim spray at fruiting stage",
         "pesticides": "Chlorothalonil 75WP @ 2g/L bi-weekly during monsoon"}]},
    "Ashwagandha": {"diseases": [
        {"name": "Leaf Blight (Alternaria)", "symptoms": "Dark brown lesions with yellow border, premature defoliation",
         "prevention": "Seed treatment, field sanitation, avoid dense planting",
         "treatment": "Spray Mancozeb at first sign, repeat every 10-15 days",
         "pesticides": "Mancozeb 75WP @ 2.5g/L, Iprodione 50WP @ 2g/L"}]},
    "Vanilla": {"diseases": [
        {"name": "Root Rot (Fusarium)", "symptoms": "Yellowing vines, black roots, collapse of climbing stems",
         "prevention": "Well-aerated support trees, mulching, avoid wetting roots",
         "treatment": "Trichoderma soil application, remove infected roots",
         "pesticides": "Carbendazim 50WP @ 1g/L drench, Pseudomonas fluorescens bioformulation"}]},
}
# Default for regions without specific data
for reg_data in REGIONS.values():
    for crop in reg_data["rare"]:
        if crop not in RARE_CROP_DISEASES:
            RARE_CROP_DISEASES[crop] = {"diseases": [
                {"name": f"Leaf Spot", "symptoms": "Spots and lesions on leaves",
                 "prevention": "Crop rotation, balanced nutrition, proper spacing",
                 "treatment": "Remove infected plant material, improve air circulation",
                 "pesticides": "Mancozeb 75WP @ 2g/L, Copper oxychloride @ 3g/L"},
                {"name": "Root Rot", "symptoms": "Wilting, yellowing, poor growth",
                 "prevention": "Raised beds, well-drained soil, avoid overwatering",
                 "treatment": "Metalaxyl soil drench, Trichoderma viride biocontrol",
                 "pesticides": "Ridomil Gold @ 2g/L soil drench"}]}

# ── Reference Data ───────────────────────────────────────────────
VAX_SCHEDULE = [
    {"disease": "Foot & Mouth Disease", "vaccine": "FMD Vaccine", "when": "4 months, then 6-monthly", "route": "Subcutaneous"},
    {"disease": "Hemorrhagic Septicemia", "vaccine": "HS Vaccine", "when": "Before monsoon, annually", "route": "Subcutaneous"},
    {"disease": "Black Quarter", "vaccine": "BQ Vaccine", "when": "Before monsoon, annually", "route": "Subcutaneous"},
    {"disease": "Brucellosis", "vaccine": "Brucella S19/RB51", "when": "Female calves 4-8 months (once)", "route": "Subcutaneous"},
    {"disease": "Theileriosis", "vaccine": "Raksha Vac T", "when": "At 2-3 months", "route": "Intramuscular"},
    {"disease": "Internal Parasites", "vaccine": "Albendazole", "when": "Every 3-4 months", "route": "Oral"},
]
DIET_REF = [
    {"component": "🌿 Green Fodder", "qty": "25–35 kg/day", "examples": "Napier, CO-4, Berseem, Lucerne, Maize"},
    {"component": "🌾 Dry Fodder", "qty": "5–8 kg/day", "examples": "Paddy straw, Wheat straw, Ragi straw"},
    {"component": "🌽 Concentrate", "qty": "1 kg per 2.5L milk + 1.5 kg", "examples": "Grain mix, compound pellets"},
    {"component": "🫘 Oil Cake", "qty": "1–2 kg/day", "examples": "Groundnut, Mustard, Cotton seed cake"},
    {"component": "💊 Mineral Mix", "qty": "50–80 g/day", "examples": "Area-specific mineral mixture"},
    {"component": "💧 Water", "qty": "50–80 L/day", "examples": "Clean, fresh, ad-libitum"},
]
MRL_DATA = [
    {"chemical": "Chlorpyrifos", "type": "Insecticide", "mrl": "0.01–0.1 mg/kg", "phi": "21-30 days", "risk": "🔴 High"},
    {"chemical": "Imidacloprid", "type": "Insecticide", "mrl": "0.05–1.0 mg/kg", "phi": "14-21 days", "risk": "🟡 Medium"},
    {"chemical": "Neem Oil", "type": "Bio-Pesticide", "mrl": "Exempt", "phi": "0-3 days", "risk": "🟢 Low"},
    {"chemical": "Mancozeb", "type": "Fungicide", "mrl": "0.05–5.0 mg/kg", "phi": "14-21 days", "risk": "🟡 Medium"},
    {"chemical": "Glyphosate", "type": "Herbicide", "mrl": "0.1–2.0 mg/kg", "phi": "30+ days", "risk": "🔴 High"},
    {"chemical": "Trichoderma", "type": "Bio-fungicide", "mrl": "Exempt", "phi": "0 days", "risk": "🟢 Low"},
]
FIRST_AID = [
    {"emergency": "🤰 Difficult Calving", "action": "Call vet immediately. Don't pull forcefully. Keep cow calm.", "time": "< 2 hrs"},
    {"emergency": "🤧 Bloat / Tympany", "action": "Keep standing. Drench 100mL vegetable oil. Trocar if severe.", "time": "< 1 hr"},
    {"emergency": "🐍 Snake Bite", "action": "Note time. Tourniquet above bite. Rush to vet for anti-venom.", "time": "< 30 min"},
    {"emergency": "🌡️ High Fever (>40.5°C)", "action": "Cold water bath. Meloxicam injection. Electrolyte water.", "time": "< 4 hrs"},
    {"emergency": "⚡ Milk Fever", "action": "IV Calcium Borogluconate (slow). Prop sternal. Monitor heart.", "time": "< 1 hr"},
]
VET_RESOURCES = [
    {"service": "Government Vet Hospital", "find": "District Collector / Block office", "coverage": "All districts"},
    {"service": "NDDB AI Services", "find": "1800-121-3456 (Toll-free)", "coverage": "Major dairy regions"},
    {"service": "KVK (Krishi Vigyan Kendra)", "find": "ICAR website → your district", "coverage": "731 KVKs nationwide"},
    {"service": "Private Vet Clinic", "find": "Google Maps / JustDial", "coverage": "Towns & cities"},
]

# ── Vector Store ─────────────────────────────────────────────────
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
            print(f"✅ Vector store: {len(self.documents)} docs")
            return True
        except Exception as e:
            print(f"⚠️  Vector store: {e}")
            return False

    def get_embedding(self, text):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model":"nomic-embed-text","prompt":text}, timeout=30)
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
        return [{"content": self.documents[idx], "metadata": self.chunks_metadata[idx], "score": float(score)}
                for score, idx in zip(scores[0], indices[0]) if idx < len(self.documents) and score > 0.1]

# ── Helpers ──────────────────────────────────────────────────────
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

def build_prompt(query, docs, farmer=None, language="English"):
    ctx = ""
    if docs:
        ctx = "\nRelevant Knowledge:\n"
        for i, d in enumerate(docs, 1):
            ctx += f"\n[Source {i}]:\n{d['content'][:400]}\n"
    farmer_ctx = ""
    if farmer:
        farmer_ctx = f"\nFarmer Context: {farmer.get('name')}, {farmer.get('region')} region, {farmer.get('land_size')} acres, grows {farmer.get('current_crop','various crops')}.\n"
    lang_instruction = f"\nRespond in {language}." if language != "English" else ""
    return f"{ctx}{farmer_ctx}{lang_instruction}\n\nFarmer's Question: {query}"

def stream_ollama_sse(model, messages, temp=0.7):
    payload = {"model": model, "messages": messages, "stream": True,
                "options": {"temperature": temp, "system": SYSTEM_PROMPT}}
    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        if "message" in chunk and "content" in chunk["message"]:
                            token = chunk["message"]["content"]
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if chunk.get("done"): break
                    except: continue
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# ── App startup ──────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

vs = VectorStoreManager(VECTOR_STORE_DIR)
vs.load_vector_store()
lm = None
if LIVESTOCK_AVAILABLE:
    try:
        lm = load_livestock_models('trained_models')
        print("✅ Livestock models loaded")
    except Exception as e: print(f"⚠️  Livestock: {e}")

# ── Schemas ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: str = "llama3.2"
    temperature: float = 0.7
    history: list = []
    farmer_profile: Optional[dict] = None
    language: str = "English"
    use_kb: bool = True

class LivestockReq(BaseModel):
    body_temp: float; heart_rate: float; respiratory_rate: float
    activity_level: float; rumination_min: float; feed_intake: float
    water_intake: float; milk_yield: float; lying_time: float
    steps_count: float; gait_score: float; stance_symmetry: float
    stride_length: float; ambient_temp: float; humidity_pct: float
    thi_index: Optional[float] = None

# ── Routes ────────────────────────────────────────────────────────
@app.get("/api/health")
@app.get("/api/status")
def health():
    return {"ollama": check_ollama(), "vector_store": vs.loaded, "livestock": True}

@app.get("/api/models")
def models():
    return {"models": get_models()}

@app.get("/api/regions")
def regions():
    return {"regions": list(REGIONS.keys())}

@app.get("/api/crops/{region}")
def crops(region: str):
    data = REGIONS.get(region, REGIONS["Western Maharashtra"])
    return {**data, "region": region}

@app.get("/api/rare-crops/{crop}")
def rare_crops(crop: str):
    data = RARE_CROP_DISEASES.get(crop, {"diseases": [
        {"name": "General Leaf Disease", "symptoms": "Spots, yellowing, wilting",
         "prevention": "Proper spacing, crop rotation, balanced nutrition",
         "treatment": "Remove infected parts, apply appropriate fungicide",
         "pesticides": "Mancozeb 75WP @ 2g/L or Copper oxychloride @ 3g/L"}]})
    return {"crop": crop, **data}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    docs = vs.search(req.message, 3) if req.use_kb and vs.loaded else []
    prompt = build_prompt(req.message, docs, req.farmer_profile, req.language)
    msgs = [{"role": m["role"], "content": m["content"]} for m in req.history[-10:]]
    msgs.append({"role": "user", "content": prompt})
    return StreamingResponse(stream_ollama_sse(req.model, msgs, req.temperature),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...), question: str = Form(...),
                         model: str = Form("llava"), language: str = Form("English")):
    try:
        img_bytes = await file.read()
        b64 = base64.b64encode(img_bytes).decode()
        lang_note = f" Respond in {language}." if language != "English" else ""
        full_question = f"{question}{lang_note}"
        payload = {"model": model, "messages": [{"role": "user", "content": full_question, "images": [b64]}],
                   "stream": False, "options": {"system": SYSTEM_PROMPT}}
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return {"analysis": r.json()["message"]["content"]}
    except Exception as e:
        return JSONResponse({"analysis": f"❌ Image analysis failed: {str(e)}. Ensure llava is installed: ollama pull llava"})

@app.post("/api/livestock/scan")
def scan(req: LivestockReq):

    if not lm:
        return {"error": "Livestock models not available"}

    try:
        data = req.dict()

        # ── Calculate THI if missing ─────────────────────────────
        if data.get("thi_index") is None:
            data["thi_index"] = round(
                0.8 * data["ambient_temp"] +
                (data["humidity_pct"] / 100) * (data["ambient_temp"] - 14.4) +
                46.4,
                1
            )

        # ── Add missing features (for full 22-feature space) ─────
        DEFAULTS = {
            "body_condition_score": 3.0,
            "age_years": 5,
            "parity": 2,
            "days_in_milk": 120,
            "previous_disease_flag": 0,
            "reproductive_status": 1
        }

        for key, value in DEFAULTS.items():
            if key not in data:
                data[key] = value

        # ── Master 22-feature list (training superset) ───────────
        master_features = [
            "body_temp", "heart_rate", "respiratory_rate",
            "activity_level", "rumination_min", "feed_intake",
            "water_intake", "milk_yield", "lying_time",
            "steps_count", "gait_score", "stance_symmetry",
            "stride_length", "ambient_temp", "humidity_pct",
            "thi_index",
            "body_condition_score",
            "age_years",
            "parity",
            "days_in_milk",
            "previous_disease_flag",
            "reproductive_status"
        ]

        df_master = pd.DataFrame(
            [[data[col] for col in master_features]],
            columns=master_features
        )

        # ── Smart prediction based on model requirement ───────────
        def smart_predict(model_obj):

            # If wrapped model
            model_core = model_obj.model if hasattr(model_obj, "model") else model_obj


             
            if hasattr(model_core, "n_features_in_"):
                required = model_core.n_features_in_
                df_input = df_master.iloc[:, :required]
            else:
        # fallback: try full dataframe
                df_input = df_master

            return model_core.predict(df_input)[0]

        health = smart_predict(lm["health_predictor"])
        disease = smart_predict(lm["disease_forecaster"])
        gait = smart_predict(lm["gait_predictor"])
        anomaly = smart_predict(lm["anomaly_detector"])

        return {
            "health": health,
            "anomaly": anomaly,
            "disease": disease,
            "gait": gait,
            "gait_cv": GaitAnalyzer().analyze_gait(data),
            "behavior": BehaviorAnalyzer().analyze_behavior(data)
        }

    except Exception as e:
        print("SCAN ERROR:", e)
        return {"error": str(e)}

@app.post("/api/livestock/scan-csv")
async def scan_csv(file: UploadFile = File(...)):
    if not lm: return {"error": "Livestock models not available"}
    try:
        import io
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        results = []
        for idx, row in df.iterrows():
            rd = row.to_dict()
            if 'thi_index' not in rd:
                rd['thi_index'] = round(0.8*rd.get('ambient_temp',30) + rd.get('humidity_pct',65)/100*(rd.get('ambient_temp',30)-14.4)+46.4,1)
            h = lm['health_predictor'].predict(rd)
            d = lm['disease_forecaster'].predict(rd)
            g = lm['gait_predictor'].predict(rd)
            a = lm['anomaly_detector'].predict(rd)
            results.append({"id": idx+1, "animal_id": rd.get('animal_id', f'A-{idx+1}'),
                "health": h['status_label'], "confidence": f"{h['confidence']*100:.0f}%",
                "disease": d['predicted_disease'].replace('_',' '),
                "gait": f"{g['gait_score']:.1f}", "anomaly": a['is_anomaly']})
        return {"total": len(results), "results": results}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/reference/vaccination")
def ref_vax(): return {"schedule": VAX_SCHEDULE}

@app.get("/api/reference/diet")
def ref_diet(): return {"diet": DIET_REF}

@app.get("/api/reference/mrl")
def ref_mrl(): return {"guidelines": MRL_DATA}

@app.get("/api/reference/first-aid")
def ref_first_aid(): return {"first_aid": FIRST_AID}

@app.get("/api/reference/vet-resources")
def ref_vet(): return {"resources": VET_RESOURCES}
@app.get("/api/weather/{region}")
def get_weather(region: str):
    return {
        "region": region,
        "forecast": [
            {
                "temperature": 32,
                "weather": "Partly Cloudy",
                "humidity": 68,
                "wind_speed": 3.2
            }
        ]
    }

# ── Static + Frontend (must be last) ─────────────────────────────
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")
