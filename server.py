import time
from test import get_audio
import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware

# Import your TTS model here
# from your_tts_module import your_tts_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/tts")
async def text_to_speech(text: str):
    start_time = time.time()
    np_audio = get_audio(text)
    print(f"Time taken for get_audio: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    buffer = io.BytesIO()
    sf.write(buffer, np_audio, samplerate=16000, format='WAV')
    print(f"Time taken for writing audio to buffer: {time.time() - start_time:.4f} seconds")
    
    buffer.seek(0)

    return StreamingResponse(
        buffer, 
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=tts_output.wav"
        }
    )