import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="FLOOR METRIC AI Architect API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "credits.db"

# Updated candidate models compatible with the current Google GenAI SDK
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash-experimental",
]


def get_genai_client() -> genai.Client:
    raw_key = (os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if not raw_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing in Render Environment Variables.",
        )
    return genai.Client(api_key=raw_key)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wallets (
                device_id TEXT PRIMARY KEY,
                has_lifetime_pass INTEGER DEFAULT 0,
                daily_credits INTEGER DEFAULT 0,
                yesterday_unused INTEGER DEFAULT 0,
                restored_bonus_credits INTEGER DEFAULT 0,
                last_reset_utc TEXT
            )
        """)


init_db()


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "FLOOR METRIC AI Architect API",
        "web_app_url": "/app/",
    }


class WallOpeningSchema(BaseModel):
    id: str
    type: Literal["doorLeft", "doorRight", "window"]
    wall: Literal["north", "south", "east", "west"]
    offsetCm: float
    widthCm: float


class CadFixtureSchema(BaseModel):
    id: str
    label: str
    symbolType: Literal["bed", "sofa", "kitchen", "bath", "desk", "wardrobe"]
    xCm: float
    yCm: float
    widthCm: float
    lengthCm: float
    rotationDeg: float


class ArchRoomSchema(BaseModel):
    id: str
    name: str
    offsetXCm: float
    offsetYCm: float
    widthCm: float
    lengthCm: float
    wallThicknessCm: float
    floorLevel: int
    openings: List[WallOpeningSchema]
    fixtures: List[CadFixtureSchema]


class HousePlanSchema(BaseModel):
    id: str
    projectName: str
    architectName: str
    rooms: List[ArchRoomSchema]


class GenerateRequest(BaseModel):
    device_id: str
    prompt: str
    floor_level: int = 0


class VerifyPurchaseRequest(BaseModel):
    device_id: str
    product_id: str
    purchase_token: str


def get_and_sync_wallet(conn, device_id: str) -> dict:
    now_utc = datetime.now(timezone.utc)
    row = conn.execute("SELECT * FROM wallets WHERE device_id = ?", (device_id,)).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO wallets VALUES (?, 1, 100, 0, 0, ?)",
            (device_id, now_utc.isoformat()),
        )
        return {
            "hasLifetimePass": True,
            "dailyCredits": 100,
            "yesterdayUnused": 0,
            "restoredBonusCredits": 0,
            "lastResetUtc": now_utc.isoformat(),
        }

    _, has_pro, daily, yest, bonus, last_reset_str = row
    last_reset = datetime.fromisoformat(last_reset_str)

    if has_pro and (now_utc - last_reset).total_seconds() >= 86400:
        yest = daily
        daily = 100
        last_reset = now_utc
        conn.execute(
            "UPDATE wallets SET daily_credits=?, yesterday_unused=?, last_reset_utc=? WHERE device_id=?",
            (daily, yest, last_reset.isoformat(), device_id),
        )

    return {
        "hasLifetimePass": bool(has_pro),
        "dailyCredits": daily,
        "yesterdayUnused": yest,
        "restoredBonusCredits": bonus,
        "lastResetUtc": last_reset.isoformat(),
    }


@app.post("/activate-pro")
def activate_pro(req: VerifyPurchaseRequest):
    now_utc = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        get_and_sync_wallet(conn, req.device_id)
        conn.execute(
            "UPDATE wallets SET has_lifetime_pass=1, daily_credits=100, last_reset_utc=? WHERE device_id=?",
            (now_utc, req.device_id),
        )
        return get_and_sync_wallet(conn, req.device_id)


@app.post("/restore-credits")
def restore_credits(req: VerifyPurchaseRequest):
    with sqlite3.connect(DB_PATH) as conn:
        wallet = get_and_sync_wallet(conn, req.device_id)
        restore_amt = wallet["yesterdayUnused"] if wallet["yesterdayUnused"] > 0 else 100
        new_bonus = wallet["restoredBonusCredits"] + restore_amt
        conn.execute(
            "UPDATE wallets SET yesterday_unused=0, restored_bonus_credits=? WHERE device_id=?",
            (new_bonus, req.device_id),
        )
        return get_and_sync_wallet(conn, req.device_id)


@app.post("/generate-plan")
def generate_plan(req: GenerateRequest):
    with sqlite3.connect(DB_PATH) as conn:
        wallet = get_and_sync_wallet(conn, req.device_id)
        if not wallet["hasLifetimePass"]:
            raise HTTPException(status_code=403, detail="Lifetime Pro Pass required.")

        total_credits = wallet["dailyCredits"] + wallet["restoredBonusCredits"]
        if total_credits < 10:
            raise HTTPException(status_code=402, detail="Insufficient credits. Restore credits (₹199).")

        system_instruction = (
            "You are a senior residential architect. Convert the prompt into a realistic, "
            "non-overlapping 2D floor plan in centimeters. Stack rooms flush edge-to-edge so adjacent "
            "rooms share walls (use wallThicknessCm=23.0, rotationDeg=0.0, "
            f"floorLevel={req.floor_level}, architectName='FLOOR METRIC AI'). "
            "Place doors on connecting walls and windows on exterior walls. "
            "Ensure all fixtures fit comfortably inside room boundaries."
        )

        client = get_genai_client()
        plan_text = None
        last_error = None

        for model_name in CANDIDATE_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"Floor Level: {req.floor_level}. Design request: {req.prompt}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=HousePlanSchema,
                        temperature=0.3,
                    ),
                )
                if response.text:
                    json.loads(response.text)
                    plan_text = response.text
                    break
            except Exception as e:
                last_error = e
                continue

        if not plan_text:
            raise HTTPException(status_code=500, detail=f"Gemini error: {str(last_error)}")

        daily = wallet["dailyCredits"]
        bonus = wallet["restoredBonusCredits"]
        if daily >= 10:
            daily -= 10
        else:
            rem = 10 - daily
            daily = 0
            bonus -= rem

        conn.execute(
            "UPDATE wallets SET daily_credits=?, restored_bonus_credits=? WHERE device_id=?",
            (daily, bonus, req.device_id),
        )
        updated_wallet = get_and_sync_wallet(conn, req.device_id)

        return {
            "wallet": updated_wallet,
            "plan_json": plan_text,
        }


if os.path.isdir("web_app"):
    app.mount("/app", StaticFiles(directory="web_app", html=True), name="web_app")
