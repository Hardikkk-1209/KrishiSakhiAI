# ═══════════════════════════════════════════════════════
# KrishiSakhiAI — FastAPI Backend (FINAL CLEAN VERSION)
# Auto-switches to LLaVA for image analysis
# ═══════════════════════════════════════════════════════

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import json
import base64

app = FastAPI()

# ─────────────────────────────────────────
# CORS
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
@app.get("/api/health")
def health_check():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_online = r.status_code == 200
    except:
        ollama_online = False

    return {
        "server": True,
        "ollama": ollama_online
    }


# ─────────────────────────────────────────
# LIST MODELS
# ─────────────────────────────────────────
@app.get("/api/models")
def get_models():
    try:
        r = requests.get("http://localhost:11434/api/tags")
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"models": models}
    except:
        return {"models": []}


# ─────────────────────────────────────────
# CHAT (STREAMING)
# ─────────────────────────────────────────
@app.post("/api/chat")
async def chat(request: dict):

    message = request.get("message")
    model = request.get("model", "llama3.2:latest")
    temperature = request.get("temperature", 0.7)
    history = request.get("history", [])

    messages = history + [{"role": "user", "content": message}]

    def stream():
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature}
                },
                stream=True
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data:
                        token = data["message"]["content"]
                        yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ─────────────────────────────────────────
# IMAGE ANALYSIS (FORCE LLAVA)
# ─────────────────────────────────────────
@app.post("/api/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = Form(...),
    language: str = Form("English")
):
    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")

        # 🔥 Strong Agricultural Prompt
        prompt = f"""
You are KrishiSakhiAI, an expert agricultural assistant.

Carefully analyze this crop image.

User Question:
{question}

Provide:
1. Disease or pest name
2. Visible symptoms
3. Possible causes
4. Treatment steps
5. Prevention methods
6. Recommended pesticide/fungicide

Respond clearly in {language}.
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",  # 🔥 FORCED MODEL
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
        )

        result = response.json()

        return {
            "analysis": result.get("response", "No analysis returned.")
        }

    except Exception as e:
        return {
            "analysis": f"Image analysis failed: {str(e)}"
        }


# ─────────────────────────────────────────
# DEMO DATA — REGIONS
# ─────────────────────────────────────────
@app.get("/api/regions")
def regions():
    return {
        "regions": ["Punjab", "Maharashtra", "Karnataka", "Tamil Nadu"]
    }


@app.get("/api/crops/{region}")
def crops(region: str):
    return {
        "climate": "Tropical / Semi-Arid",
        "soil": "Loamy / Black Soil",
        "common": ["Wheat", "Rice", "Maize"],
        "rare": ["Dragon Fruit", "Quinoa"]
    }


@app.get("/api/rare-crops/{crop}")
def rare_crop(crop: str):
    return {
        "diseases": [
            {
                "name": "Leaf Spot",
                "symptoms": "Yellow or brown spots on leaves.",
                "prevention": "Use disease-resistant varieties.",
                "treatment": "Apply copper-based fungicide.",
                "pesticides": "Mancozeb"
            }
        ]
    }