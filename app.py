import os
import tempfile
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

import constants

transcribe_pipeline: AutomaticSpeechRecognitionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcribe_pipeline
    print("Loading Whisper model...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    transcribe_pipeline = pipeline(
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
app.mount("/static", StaticFiles(directory=constants.STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(constants.STATIC_DIR, "index.html"))


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if transcribe_pipeline is None:
        return JSONResponse(status_code=503, content={"detail": "Model not ready"})

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = transcribe_pipeline(tmp_path)
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
