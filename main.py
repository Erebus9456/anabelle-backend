from colab_compat import apply_runtime_patches

apply_runtime_patches()

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from engine import AnabelleEngine
import numpy as np
import json
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnabelleGateway")

app = FastAPI()

# Initialize the engine once on startup
engine = AnabelleEngine()

@app.websocket("/ws/anabelle")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to ANABELLE Gateway.")
    
    try:
        while True:
            # Receive binary audio data from the React app
            # Audio is expected to be Float32 PCM at 16kHz
            data = await websocket.receive_bytes()
            
            # Convert binary bytes back to a NumPy array for the engine
            audio_array = np.frombuffer(data, dtype=np.float32)
            
            # Run the AI Inference
            # SenseVoice performs best on chunks of 0.5s - 2s
            result = engine.analyze_chunk(audio_array)
            logger.info("Engine Decision: %s", result["emotion"])
            await websocket.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Gateway Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    # Start the server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
    