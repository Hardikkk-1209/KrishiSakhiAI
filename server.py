"""
KrishiSakhiAI — FastAPI Backend Server
Replaces Streamlit; serves HTML/CSS/JS frontend + REST API
"""

import os, sys, json, base64, requests, pickle, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from livestock_biosecurity.models import load_models as load_livestock_models, HEALTH_LABELS
    from livestock_biosecurity.cv_module import GaitAnalyzer, BehaviorAnalyzer
    LIVESTOCK_AVAILABLE = True
except ImportError:
    LIVESTOCK_AVAILABLE = False

# ── Config ──
OLLAMA_URL = "http://localhost:11434"
VECTOR_STORE_DIR = "vector_store"

# ══════════════════════════════════════════
# CROP & DISEASE DATA (same as Streamlit)
# ══════════════════════════════════════════

MAHARASHTRA_CROPS = {
    "Nashik": {"common":["Grapes","Onion","Pomegranate","Tomato","Wheat","Bajra","Sugarcane","Maize"],"rare":["Saffron","Avocado"],"climate":"Semi-arid, moderate rainfall","soil":"Black soil, Medium-deep soils"},
    "Pune": {"common":["Sugarcane","Rice","Wheat","Jowar","Bajra","Vegetables","Flowers (Rose, Marigold)","Turmeric"],"rare":["Avocado"],"climate":"Tropical wet & dry, moderate rainfall","soil":"Red laterite, Black soil"},
    "Konkan": {"common":["Rice","Mango (Alphonso)","Cashew","Coconut","Kokum","Jackfruit","Betel Nut","Spices"],"rare":["Coffee"],"climate":"Tropical, heavy monsoon","soil":"Laterite, Coastal alluvial"},
    "Vidarbha": {"common":["Cotton","Soybean","Orange (Nagpur)","Jowar","Tur Dal","Wheat","Chilli","Sunflower"],"rare":["Dragon Fruit"],"climate":"Hot semi-arid, moderate rainfall","soil":"Black cotton soil, Deep clay"},
    "Ahmednagar": {"common":["Sugarcane","Onion","Pomegranate","Bajra","Groundnut","Sunflower","Milk production","Jowar"],"rare":["Olive"],"climate":"Semi-arid, drought-prone areas","soil":"Shallow to medium black soil"},
    "Mahabaleshwar": {"common":["Strawberry","Raspberry","Mulberry","Carrot","Beetroot","Peas","Potato","Leafy vegetables"],"rare":["Blueberry"],"climate":"Hill station, high rainfall, cool climate","soil":"Laterite, Rich humus forest soil"},
    "Solapur": {"common":["Pomegranate","Sugarcane","Jowar","Bajra","Tur Dal","Grapes","Onion","Groundnut"],"rare":["Pistachio"],"climate":"Hot semi-arid, low rainfall, drought-prone","soil":"Shallow black soil, Rocky terrain"},
    "Satara": {"common":["Sugarcane","Rice","Turmeric","Strawberry","Potato","Wheat","Jowar","Vegetables"],"rare":["Kiwi"],"climate":"Moderate, good rainfall in western parts","soil":"Red laterite, Black soil"},
}

RARE_CROP_DISEASES = {
    "Saffron":[
        {"name":"Corm Rot (Fusarium oxysporum)","symptoms":"Yellowing and wilting of leaves, soft brown rot on corms, foul smell from infected bulbs, stunted growth","prevention":"Use certified disease-free corms, practice crop rotation (3-4 years), ensure well-drained soil, avoid waterlogging","treatment":"Remove and destroy infected corms immediately, apply Carbendazim (Bavistin) 2g/L as soil drench","pesticides":"Carbendazim 50% WP — 2g/L; Copper Oxychloride — 3g/L; Trichoderma viride — 5g/kg corm"},
        {"name":"Leaf Blight (Rhizoctonia crocinum)","symptoms":"Dark brown to black spots on leaves, leaves dry from tips downward, reduced flower production","prevention":"Avoid overhead irrigation, maintain proper spacing, remove crop debris after harvest","treatment":"Spray Mancozeb 75% WP at 2.5g/L at first symptom, repeat every 10-14 days","pesticides":"Mancozeb 75% WP — 2.5g/L; Chlorothalonil — 2g/L; Hexaconazole 5% EC — 1ml/L"},
    ],
    "Avocado":[
        {"name":"Phytophthora Root Rot","symptoms":"Wilting despite adequate water, small pale-green leaves, branch dieback, dark brown/black roots, fruit drop","prevention":"Plant in well-drained soil, use resistant rootstocks (Dusa, Latas), avoid overwatering, mulch","treatment":"Apply Metalaxyl (Ridomil Gold) as soil drench, inject trunk with Phosphonate (Aliette)","pesticides":"Metalaxyl 4% GR — 25g/tree; Fosetyl-Al (Aliette) 80% WP — 2.5g/L; Potassium Phosphonate — trunk injection"},
        {"name":"Anthracnose (Colletotrichum)","symptoms":"Dark circular lesions on fruits, black spots on leaves, fruit turns black during ripening","prevention":"Prune for air circulation, harvest at correct maturity, pre-harvest fungicide sprays","treatment":"Spray Azoxystrobin at flowering, post-harvest hot water treatment (48°C, 20 min)","pesticides":"Azoxystrobin 23% SC — 1ml/L; Copper Hydroxide — 2g/L; Carbendazim — 1g/L"},
        {"name":"Cercospora Leaf Spot","symptoms":"Angular brown spots on leaves with yellow halos, premature leaf drop","prevention":"Maintain tree vigor with balanced fertilization, remove fallen leaves","treatment":"Apply Copper-based fungicides, spray Mancozeb at 2-week intervals during wet season","pesticides":"Mancozeb 75% WP — 2.5g/L; Copper Oxychloride — 3g/L; Thiophanate-methyl — 1g/L"},
    ],
    "Coffee":[
        {"name":"Coffee Leaf Rust (Hemileia vastatrix)","symptoms":"Orange-yellow powdery spots on leaf undersides, premature leaf fall, 30-80% yield loss","prevention":"Plant rust-resistant varieties (Sln.5, Sln.9), shade management, balanced nutrition","treatment":"Spray Bordeaux mixture (1%) pre-monsoon, apply systemic fungicides at first sign","pesticides":"Bordeaux Mixture 1%; Tridemorph (Calixin) 80% EC — 0.5ml/L; Hexaconazole 5% EC — 2ml/L"},
        {"name":"Coffee Berry Disease","symptoms":"Dark sunken lesions on green berries, mummified black berries, premature berry drop","prevention":"Maintain shade canopy, prune for airflow, remove mummified berries","treatment":"Spray Copper Oxychloride at pea-berry stage, Carbendazim every 3 weeks","pesticides":"Copper Oxychloride 50% WP — 3g/L; Carbendazim 50% WP — 1g/L; Chlorothalonil — 2g/L"},
    ],
    "Dragon Fruit":[
        {"name":"Stem Rot (Fusarium)","symptoms":"Water-soaked soft spots on stems, yellowing at base, brown mushy rot spreading upward","prevention":"Avoid overhead irrigation, ensure drainage, sterilized cutting tools","treatment":"Cut infected parts 2 inches below infection, apply Bordeaux paste, drench with Copper Oxychloride","pesticides":"Bordeaux Paste; Copper Oxychloride 50% WP — 3g/L; Metalaxyl + Mancozeb — 2.5g/L; Trichoderma"},
        {"name":"Anthracnose (Colletotrichum sp.)","symptoms":"Brown sunken spots on fruit, cracking of skin, reduced shelf life","prevention":"Pre-harvest sprays at flowering, avoid fruit injury, harvest at right maturity","treatment":"Spray Azoxystrobin at flowering, Mancozeb every 14 days during fruit development","pesticides":"Azoxystrobin 23% SC — 1ml/L; Mancozeb 75% WP — 2.5g/L; Carbendazim — 1g/L"},
    ],
    "Olive":[
        {"name":"Olive Knot (Pseudomonas savastanoi)","symptoms":"Rough woody galls on branches, reduced fruiting, branch dieback","prevention":"Prune during dry weather, sterilize tools between cuts, use resistant varieties","treatment":"Prune all knotted branches 10cm below galls, apply Copper spray after pruning","pesticides":"Copper Hydroxide 77% WP — 2.5g/L; Bordeaux Mixture 1%; Streptomycin — 0.5g/L"},
        {"name":"Olive Leaf Spot","symptoms":"Dark circular spots on upper leaf surface, yellow halo, premature leaf fall","prevention":"Good air circulation through pruning, balanced fertilization","treatment":"Spray Copper-based fungicides before monsoon, Dodine in severe cases","pesticides":"Copper Oxychloride — 3g/L; Dodine 65% WP — 1g/L; Mancozeb — 2.5g/L"},
    ],
    "Blueberry":[
        {"name":"Mummy Berry","symptoms":"Flowers turn brown, berries shrivel into hard mummies, gray spore masses","prevention":"Remove mummified berries, 2-inch mulch layer, good air circulation","treatment":"Spray Propiconazole at bud break, Chlorothalonil at bloom","pesticides":"Propiconazole 25% EC — 1ml/L; Chlorothalonil 75% WP — 2g/L; Azoxystrobin — 1ml/L"},
        {"name":"Botrytis Gray Mold","symptoms":"Gray fuzzy mold on berries, blossom blight, soft watery decay","prevention":"Harvest frequently, improve airflow, avoid overhead irrigation","treatment":"Spray Iprodione or Fenhexamid at 10% bloom, rotate fungicide groups","pesticides":"Iprodione (Rovral) 50% WP — 2g/L; Fenhexamid (Elevate) — 1.5g/L; Switch — 0.8g/L"},
    ],
    "Pistachio":[
        {"name":"Alternaria Late Blight","symptoms":"Black lesions on leaves and nuts, staining of shells, premature leaf fall","prevention":"Open canopy, proper irrigation, balanced nitrogen","treatment":"Spray Azoxystrobin at shell hardening, repeat every 14 days","pesticides":"Azoxystrobin 23% SC — 1ml/L; Pyraclostrobin — 0.5ml/L; Copper Hydroxide — 2.5g/L"},
        {"name":"Botryosphaeria Blight","symptoms":"Dark killed flower clusters, shoot dieback, cankers on branches","prevention":"Remove dead wood, avoid sprinkler wetting canopy, prune for air circulation","treatment":"Prune blighted shoots, apply Thiophanate-methyl at bud swell","pesticides":"Thiophanate-methyl 70% WP — 1g/L; Pyraclostrobin + Boscalid — 0.5g/L"},
    ],
    "Kiwi":[
        {"name":"Bacterial Canker (PSA)","symptoms":"Reddish-brown cankers with ooze on trunks, leaf spots with yellow halos, vine death","prevention":"Use PSA-resistant varieties, sterilize tools, avoid wet-weather pruning","treatment":"Cut cankers 30cm below infection, Copper sprays 3-4 times during dormancy","pesticides":"Copper Hydroxide 77% WP — 2.5g/L; Kasugamycin — 2ml/L; Streptomycin — 0.5g/L"},
        {"name":"Botrytis Fruit Rot","symptoms":"Soft watery rot at stem end, gray fluffy mold, rapid post-harvest decay","prevention":"Avoid fruit injury, harvest at 6.2% Brix, cold storage at 0°C immediately","treatment":"Spray Iprodione pre-harvest, dip fruit in Fludioxonil solution","pesticides":"Iprodione — 2g/L; Fenhexamid — 1.5g/L; Fludioxonil — post-harvest dip"},
    ],
}

VACCINATION_SCHEDULE = [
    {"disease":"Foot & Mouth Disease (FMD)","vaccine":"FMD Vaccine","when":"At 4 months, then every 6 months","route":"Subcutaneous"},
    {"disease":"Hemorrhagic Septicemia (HS)","vaccine":"HS Vaccine","when":"Before monsoon, annually","route":"Subcutaneous"},
    {"disease":"Black Quarter (BQ)","vaccine":"BQ Vaccine","when":"Before monsoon, annually","route":"Subcutaneous"},
    {"disease":"Brucellosis","vaccine":"Brucella S19 / RB51","when":"Female calves 4-8 months (once)","route":"Subcutaneous"},
    {"disease":"Theileriosis","vaccine":"Raksha Vac T","when":"At 2-3 months","route":"Intramuscular"},
    {"disease":"Anthrax","vaccine":"Anthrax Spore Vaccine","when":"Annually in endemic areas","route":"Subcutaneous"},
    {"disease":"Internal Parasites","vaccine":"Albendazole / Fenbendazole","when":"Every 3-4 months","route":"Oral"},
    {"disease":"External Parasites","vaccine":"Ivermectin / Deltamethrin","when":"Every 2-3 months","route":"Pour-on / Injection"},
]

RECOMMENDED_DIET = [
    {"component":"🌿 Green Fodder","qty":"25–35 kg/day","examples":"Napier, CO-4, Berseem, Lucerne, Maize"},
    {"component":"🌾 Dry Fodder","qty":"5–8 kg/day","examples":"Paddy straw, Wheat straw, Ragi straw"},
    {"component":"🌽 Concentrate","qty":"1 kg per 2.5L milk + 1.5 kg","examples":"Grain mix, compound pellets"},
    {"component":"🫘 Oil Cake","qty":"1–2 kg/day","examples":"Groundnut, Mustard, Cotton seed cake"},
    {"component":"💊 Mineral Mix","qty":"50–80 g/day","examples":"Area-specific mineral mixture"},
    {"component":"🧂 Salt","qty":"25–35 g/day","examples":"Rock salt / iodised salt"},
    {"component":"💧 Water","qty":"50–80 L/day","examples":"Clean, fresh, ad-libitum"},
]

MRL_GUIDELINES = [
    {"chemical":"Chlorpyrifos","type":"Insecticide","mrl":"0.01–0.1","phi":"21-30 days","risk":"High"},
    {"chemical":"Imidacloprid","type":"Insecticide","mrl":"0.05–1.0","phi":"14-21 days","risk":"Medium"},
    {"chemical":"Neem Oil","type":"Bio-Pesticide","mrl":"Exempt","phi":"0-3 days","risk":"Low"},
    {"chemical":"Mancozeb","type":"Fungicide","mrl":"0.05–5.0","phi":"14-21 days","risk":"Medium"},
    {"chemical":"Glyphosate","type":"Herbicide","mrl":"0.1–2.0","phi":"30+ days","risk":"High"},
    {"chemical":"Trichoderma","type":"Bio-fungicide","mrl":"Exempt","phi":"0 days","risk":"Low"},
]

FIRST_AID = [
    {"emergency":"Difficult Calving","action":"Call vet immediately. Don't pull forcefully. Keep cow calm.","time":"< 2 hrs"},
    {"emergency":"Bloat / Tympany","action":"Keep standing. Drench 100mL vegetable oil. Trocar if severe.","time":"< 1 hr"},
    {"emergency":"Snake Bite","action":"Note time. Tourniquet above bite. Rush to vet for anti-venom.","time":"< 30 min"},
    {"emergency":"Poisoning","action":"Identify poison. Activated charcoal (1-3 g/kg BW). Call vet.","time":"< 1 hr"},
    {"emergency":"Heavy Bleeding","action":"Apply pressure with clean cloth. Elevate. Keep calm.","time":"Immediate"},
    {"emergency":"High Fever (>40.5°C)","action":"Cold water bath. Meloxicam injection. Electrolyte water.","time":"< 4 hrs"},
    {"emergency":"Fracture / Down","action":"Do NOT lift. Immobilize limb. Provide bedding. Call vet.","time":"< 6 hrs"},
    {"emergency":"Milk Fever","action":"IV Calcium Borogluconate (slow). Prop sternal. Monitor heart.","time":"< 1 hr"},
]

VET_RESOURCES = [
    {"service":"Government Vet Hospital","find":"District Collector / Block office","coverage":"All districts"},
    {"service":"NDDB AI Services","find":"1800-121-3456","coverage":"Major dairy regions"},
    {"service":"Mobile Vet Unit (MVU)","find":"State Animal Husbandry Dept.","coverage":"Select blocks"},
    {"service":"KVK (Krishi Vigyan Kendra)","find":"ICAR website → your district","coverage":"731 KVKs nationwide"},
    {"service":"Veterinary College","find":"Nearest DUVASU / state vet university","coverage":"State capitals"},
    {"service":"Private Vet Clinic","find":"Google Maps / JustDial","coverage":"Towns & cities"},
]


# ══════════════════════════════════════════
# LOAD ML MODELS
# ══════════════════════════════════════════
livestock_models = None
if LIVESTOCK_AVAILABLE:
    try:
        livestock_models = load_livestock_models('trained_models')
        print(f"   ✅ Loaded {len(livestock_models)} livestock ML models")
    except Exception as e:
        print(f"   ⚠️ Could not load models: {e}")

# ── Vector Store (simple) ──
vector_store_data = None
try:
    import faiss
    idx_path = Path(VECTOR_STORE_DIR) / "faiss_index.bin"
    meta_path = Path(VECTOR_STORE_DIR) / "metadata.pkl"
    if idx_path.exists() and meta_path.exists():
        faiss_index = faiss.read_index(str(idx_path))
        with open(meta_path, 'rb') as f:
            vs_metadata = pickle.load(f)
        vector_store_data = {"index": faiss_index, "metadata": vs_metadata}
        print(f"   ✅ Vector store loaded ({faiss_index.ntotal} vectors)")
except Exception as e:
    print(f"   ⚠️ Vector store not available: {e}")


# ══════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════
app = FastAPI(title="KrishiSakhiAI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic Models ──
class FarmerProfile(BaseModel):
    name: str
    phone: str
    land_size: float = 5.0
    water_source: str = "Well / Borewell"
    current_crop: str = ""
    region: str = "Nashik"

class ChatRequest(BaseModel):
    message: str
    model: str = "llama3.2:1b"
    temperature: float = 0.7
    history: list = []
    farmer_profile: Optional[dict] = None
    language: str = "English"

LANG_INSTRUCTIONS = {
    "English": "",
    "Hindi": "\n\nIMPORTANT: You MUST respond entirely in Hindi (Devanagari script). Use Hindi for all explanations, advice, and responses. Example: 'नमस्ते किसान भाई!'",
    "Marathi": "\n\nIMPORTANT: You MUST respond entirely in Marathi (Devanagari script). Use Marathi for all explanations, advice, and responses. Example: 'नमस्कार शेतकरी मित्रा!'",
}

class LivestockReading(BaseModel):
    body_temp: float = 38.8
    heart_rate: int = 60
    respiratory_rate: int = 22
    activity_level: int = 68
    rumination_min: int = 450
    feed_intake: float = 22.0
    water_intake: float = 55.0
    milk_yield: float = 20.0
    lying_time: float = 12.0
    steps_count: int = 3500
    gait_score: float = 1.2
    stance_symmetry: float = 0.95
    stride_length: float = 1.5
    ambient_temp: float = 30.0
    humidity_pct: float = 65.0
    thi_index: float = 0.0


# ══════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health():
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except: pass
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "livestock_models": livestock_models is not None,
        "vector_store": vector_store_data is not None,
    }

@app.get("/api/models")
async def get_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return {"models": [m["name"] for m in r.json().get("models", [])]}
    except: pass
    return {"models": []}

# ── Crop Data ──
@app.get("/api/regions")
async def get_regions():
    return {"regions": list(MAHARASHTRA_CROPS.keys())}

@app.get("/api/crops/{region}")
async def get_crops(region: str):
    data = MAHARASHTRA_CROPS.get(region)
    if not data:
        raise HTTPException(404, f"Region '{region}' not found")
    return data

@app.get("/api/rare-crops/{crop}")
async def get_rare_crop_diseases(crop: str):
    diseases = RARE_CROP_DISEASES.get(crop, [])
    return {"crop": crop, "diseases": diseases}

# ── Reference Data ──
@app.get("/api/reference/vaccination")
async def get_vaccination_schedule():
    return {"schedule": VACCINATION_SCHEDULE}

@app.get("/api/reference/diet")
async def get_recommended_diet():
    return {"diet": RECOMMENDED_DIET}

@app.get("/api/reference/mrl")
async def get_mrl_guidelines():
    return {"guidelines": MRL_GUIDELINES}

@app.get("/api/reference/first-aid")
async def get_first_aid():
    return {"first_aid": FIRST_AID}

@app.get("/api/reference/vet-resources")
async def get_vet_resources():
    return {"resources": VET_RESOURCES}

# ── Chat (Streaming) ──
def build_system_prompt(farmer_profile: dict = None, language: str = "English"):
    base = """You are KrishiSakhi, an advanced AI agricultural assistant with deep expertise in farming practices.

You provide:
✅ Practical, actionable advice for crop management
✅ Pest and disease identification and treatment
✅ Weather-based farming recommendations
✅ Sustainable and organic farming methods
✅ Cost-effective solutions for small and large farmers
✅ Soil health and irrigation guidance
✅ Market trends and crop economics

Always prioritize: farmer safety, sustainability, cost-effectiveness, local conditions, simplicity.
Communicate warmly using clear, practical language."""

    if farmer_profile:
        region = farmer_profile.get('region', 'Unknown')
        region_info = MAHARASHTRA_CROPS.get(region, {})
        rare = ', '.join(region_info.get('rare', []))
        common = ', '.join(region_info.get('common', []))
        base += f"""

--- FARMER PROFILE ---
Name: {farmer_profile.get('name', 'Farmer')}
Region: {region}, Maharashtra
Land: {farmer_profile.get('land_size', 'N/A')} acres
Water: {farmer_profile.get('water_source', 'N/A')}
Current Crops: {farmer_profile.get('current_crop', 'N/A')}
Climate: {region_info.get('climate', 'N/A')}
Soil: {region_info.get('soil', 'N/A')}
Common Crops: {common}
Rare Crop Opportunities: {rare}

Tailor ALL answers to this farmer's region, soil, climate, water source, and land size. Address by name when appropriate."""

    # Add language instruction
    base += LANG_INSTRUCTIONS.get(language, "")
    return base


@app.post("/api/chat")
async def chat(req: ChatRequest):
    system_prompt = build_system_prompt(req.farmer_profile, req.language)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.history[-10:]:  # last 10 messages for context
        messages.append(msg)
    messages.append({"role": "user", "content": req.message})

    def stream():
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": req.model, "messages": messages, "stream": True,
                      "options": {"temperature": req.temperature}},
                stream=True, timeout=120
            )
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if data.get("done"):
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Image Upload & Analysis ──
@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...), question: str = "What crop disease or pest do you see? Provide diagnosis, treatment, and prevention.", model: str = "llava", language: str = "English"):
    """Analyze a crop/livestock image using Ollama vision model"""
    image_bytes = await file.read()
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')

    lang_note = LANG_INSTRUCTIONS.get(language, "")
    full_question = f"""You are KrishiSakhi, an expert agricultural AI assistant. Analyze this image carefully.

{question}

Provide your response in this format:
1. **Identification**: What do you see? (crop, disease, pest, soil issue, livestock condition)
2. **Diagnosis**: What is the likely problem?
3. **Symptoms observed**: Describe visible symptoms
4. **Treatment**: Recommend specific treatments and pesticides/fungicides with dosage
5. **Prevention**: How to prevent this in the future
6. **Urgency**: Rate urgency (Low / Medium / High / Critical)
{lang_note}"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": full_question, "images": [img_b64]}],
                  "stream": False, "options": {"temperature": 0.3}},
            timeout=120
        )
        if resp.status_code == 200:
            result = resp.json()
            answer = result.get("message", {}).get("content", "Could not analyze the image.")
            return {"analysis": answer, "model_used": model}
        else:
            return {"analysis": "⚠️ Vision model (llava) not available. Install with: ollama pull llava", "model_used": "none"}
    except Exception as e:
        raise HTTPException(500, f"Image analysis failed: {str(e)}")


# ── Livestock Scan ──
@app.post("/api/livestock/scan")
async def livestock_scan(reading: LivestockReading):
    if not livestock_models:
        raise HTTPException(503, "Livestock models not loaded")

    rd = reading.model_dump()
    rd['thi_index'] = round(0.8 * rd['ambient_temp'] + rd['humidity_pct'] / 100 * (rd['ambient_temp'] - 14.4) + 46.4, 1)

    health = livestock_models['health_predictor'].predict(rd)
    anomaly = livestock_models['anomaly_detector'].predict(rd)
    gait = livestock_models['gait_predictor'].predict(rd)
    disease = livestock_models['disease_forecaster'].predict(rd)

    gait_analyzer = GaitAnalyzer()
    gait_cv = gait_analyzer.analyze_gait(rd)
    behavior_analyzer = BehaviorAnalyzer()
    behavior = behavior_analyzer.analyze_behavior(rd)

    return {
        "health": health,
        "anomaly": anomaly,
        "gait": gait,
        "disease": disease,
        "gait_cv": gait_cv,
        "behavior": behavior,
        "reading": rd,
    }

@app.post("/api/livestock/scan-csv")
async def livestock_scan_csv(file: UploadFile = File(...)):
    if not livestock_models:
        raise HTTPException(503, "Livestock models not loaded")
    try:
        df = pd.read_csv(file.file)
        results = []
        for idx, row in df.iterrows():
            rd = row.to_dict()
            h = livestock_models['health_predictor'].predict(rd)
            d = livestock_models['disease_forecaster'].predict(rd)
            g = livestock_models['gait_predictor'].predict(rd)
            a = livestock_models['anomaly_detector'].predict(rd)
            results.append({
                'id': idx + 1,
                'animal_id': rd.get('animal_id', f'A-{idx+1}'),
                'health': h['status_label'],
                'confidence': f"{h['confidence']*100:.0f}%",
                'disease': d['predicted_disease'].replace('_', ' '),
                'gait': f"{g['gait_score']:.1f}",
                'anomaly': a['is_anomaly'],
            })
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(400, str(e))


# ══════════════════════════════════════════
# RUN
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("\n🌾 KrishiSakhiAI Server Starting...")
    print(f"   🔗 Ollama: {OLLAMA_URL}")
    print(f"   🐄 Livestock Models: {'✅ Loaded' if livestock_models else '❌ Not available'}")
    print(f"   📚 Vector Store: {'✅ Ready' if vector_store_data else '❌ Not loaded'}")
    print(f"\n   🌐 Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
