from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# COLOCAR ESTO ANTES DE CUALQUIER RUTA O ROUTER
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://optimatradingv2-seven.vercel.app"],  # Origen exacto de tu imagen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/engine/status")
async def get_status():
    return {"status": "active", "engine": "Optima V2"}
