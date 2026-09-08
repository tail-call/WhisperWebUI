import os
import tempfile
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print("Loading Whisper model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )
    print("Whisper model loaded.")
    yield


app = FastAPI(title="Whisper ASR API", lifespan=lifespan)

# Serve static frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if pipe is None:
        return JSONResponse(status_code=503, content={"detail": "Model not ready"})

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipe(tmp_path)
        chunks = result.get("chunks", [])
        return {
            "text": result["text"],
            "chunks": chunks,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run makes this thing runnable via `python main.py`
    # normally you omit the `if __name__...` part and use
    # `fastapi dev main.py` to run the app with auto-reloading
    uvicorn.run(app, host="0.0.0.0", port=8009)
