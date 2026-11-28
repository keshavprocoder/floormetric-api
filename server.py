import hashlib
import hmac
import os
import re
import secrets
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

import jwt
import psycopg2
import requests
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
JWT_ALGORITHM = "HS256"
OTP_EXPIRY_MINUTES = 10
JWT_EXPIRY_DAYS = 365

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET environment variable is missing.")
    return secret


@contextmanager
def get_db():
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is missing.")
    if "sslmode=" not in db_url:
        sep = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{sep}sslmode=require"
    conn = psycopg2.connect(db_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    has_lifetime_pass BOOLEAN NOT NULL DEFAULT FALSE,
                    daily_credits INTEGER NOT NULL DEFAULT 0,
                    yesterday_unused INTEGER NOT NULL DEFAULT 0,
                    restored_bonus_credits INTEGER NOT NULL DEFAULT 0,
                    last_reset_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_restore_utc TIMESTAMPTZ;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS email_otps (
                    email TEXT PRIMARY KEY,
                    otp_hash TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="FLOOR METRIC API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────
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
    rotationDeg: float = 0.0


class ArchRoomSchema(BaseModel):
    id: str
    name: str
    offsetXCm: float
    offsetYCm: float
    widthCm: float
    lengthCm: float
    wallThicknessCm: float = 23.0
    floorLevel: int = 0
    openings: List[WallOpeningSchema]
    fixtures: List[CadFixtureSchema]


class HousePlanSchema(BaseModel):
    id: str
    projectName: str
    architectName: str = "FLOOR METRIC AI STUDIO"
    rooms: List[ArchRoomSchema]


class SendOtpRequest(BaseModel):
    email: str


class VerifyOtpRequest(BaseModel):
    email: str
    otp: str


class GenerateRequest(BaseModel):
    prompt: str
    floor_level: int = 0


class VerifyPurchaseRequest(BaseModel):
    product_id: str
    purchase_token: str


# ── Auth & Email Helpers ─────────────────────────────────────────
def _normalize_email(raw: str) -> str:
    email = raw.strip().lower()
    if not EMAIL_REGEX.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    return email


def _hash_otp(email: str, otp: str) -> str:
    secret = _get_jwt_secret().encode("utf-8")
    msg = f"{email}:{otp}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def _create_jwt(email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=JWT_EXPIRY_DAYS)).timestamp()),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def get_current_email(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Login required.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid session token.")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token.")


def _send_otp_email(to_email: str, otp: str):
    subject = "Floor Metric - Sign-In Verification Code"
    text_body = f"Your Floor Metric verification code is: {otp}\n\nThis code expires in {OTP_EXPIRY_MINUTES} minutes. If you did not request this code, please ignore this message."
    html_body = f"""<!DOCTYPE html><html><body style="margin:0;padding:20px;background:#f8fafc;font-family:Arial,Helvetica,sans-serif;"><div style="max-width:460px;margin:auto;background:#ffffff;padding:24px;border:1px solid #e2e8f0;border-radius:12px;"><h2 style="margin:0 0 12px;color:#0f172a;font-size:20px;">Floor Metric Verification</h2><p style="color:#334155;font-size:14px;line-height:1.5;">Use the 6-digit verification code below to sign in to your Floor Metric account. This code expires in {OTP_EXPIRY_MINUTES} minutes.</p><div style="font-size:30px;font-weight:bold;letter-spacing:8px;color:#0284c7;background:#f0f9ff;padding:16px;text-align:center;border-radius:8px;margin:20px 0;">{otp}</div><p style="color:#64748b;font-size:12px;line-height:1.4;">If you did not request this code, you can safely ignore this email.</p></div></body></html>"""
    gas_url = os.environ.get("GAS_EMAIL_URL", "").strip()
    if gas_url:
        resp = requests.post(gas_url, json={"secret": os.environ.get("GAS_EMAIL_SECRET", "").strip(), "to": to_email, "subject": subject, "text": text_body, "html": html_body}, timeout=15)
        if resp.status_code >= 400:
            raise HTTPException(status_code=400, detail="Failed to send verification email.")
        return
    brevo_key = os.environ.get("BREVO_API_KEY", "").strip()
    sender_email = os.environ.get("SENDER_EMAIL", "").strip()
    if brevo_key and sender_email:
        resp = requests.post("https://api.brevo.com/v3/smtp/email", headers={"api-key": brevo_key, "Content-Type": "application/json", "accept": "application/json"}, json={"sender": {"name": "Floor Metric", "email": sender_email}, "replyTo": {"name": "Floor Metric Support", "email": sender_email}, "to": [{"email": to_email}], "subject": subject, "textContent": text_body, "htmlContent": html_body}, timeout=15)
        if resp.status_code >= 400:
            raise HTTPException(status_code=400, detail="Failed to send verification email via Brevo.")
        return
    raise HTTPException(status_code=500, detail="Email service not configured. Add GAS_EMAIL_URL or BREVO_API_KEY in Render.")

# ── Wallet Helpers (Supabase PostgreSQL) ─────────────────────────
def get_and_sync_wallet(conn, email: str) -> dict:
    now_utc = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT has_lifetime_pass, daily_credits, yesterday_unused,
                   restored_bonus_credits, last_reset_utc, last_restore_utc
            FROM users
            WHERE email = %s
            """,
            (email,),
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                """
                INSERT INTO users (email, has_lifetime_pass, daily_credits,
                                   yesterday_unused, restored_bonus_credits, last_reset_utc)
                VALUES (%s, FALSE, 0, 0, 0, %s)
                """,
                (email, now_utc),
            )
            return {
                "hasLifetimePass": False,
                "dailyCredits": 0,
                "yesterdayUnused": 0,
                "restoredBonusCredits": 0,
                "lastResetUtc": now_utc.isoformat(),
                "lastRestoreUtc": None,
            }

        has_pro, daily, yest, bonus, last_reset, last_restore = row
        if bonus > 100:
            bonus = 100
            cur.execute("UPDATE users SET restored_bonus_credits = 100 WHERE email = %s", (email,))
        if last_restore and last_restore.tzinfo is None:
            last_restore = last_restore.replace(tzinfo=timezone.utc)
        if has_pro and last_restore is None:
            last_restore = now_utc
            cur.execute("UPDATE users SET last_restore_utc = %s WHERE email = %s", (last_restore, email))
        if last_reset.tzinfo is None:
            last_reset = last_reset.replace(tzinfo=timezone.utc)

        if has_pro and (now_utc - last_reset).total_seconds() >= 86400:
            yest = daily
            daily = 100
            last_reset = now_utc
            cur.execute(
                """
                UPDATE users
                SET daily_credits = %s, yesterday_unused = %s, last_reset_utc = %s
                WHERE email = %s
                """,
                (daily, yest, last_reset, email),
            )

        return {
            "hasLifetimePass": bool(has_pro),
            "dailyCredits": daily,
            "yesterdayUnused": yest,
            "restoredBonusCredits": bonus,
            "lastResetUtc": last_reset.isoformat(),
            "lastRestoreUtc": last_restore.isoformat() if last_restore else None,
        }


# ── Public Landing, Privacy Policy & Delete Account Pages ────────
@app.get("/", response_class=HTMLResponse)
def landing_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <title>Floor Metric — Architectural CAD & 3D Studio</title>
      <style>
        body{margin:0;font-family:system-ui,sans-serif;background:#0a192f;color:#f8fafc;display:flex;min-height:100vh;align-items:center;justify-content:center;padding:24px;}
        .card{max-width:520px;background:#112240;border:1px solid #1e3a5f;border-radius:18px;padding:32px;text-align:center;}
        h1{margin:0 0 10px;color:#38bdf8;font-size:28px;}
        p{color:#cbd5e1;line-height:1.6;}
        a{color:#7dd3fc;text-decoration:none;margin:0 10px;font-size:14px;}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>FLOOR METRIC</h1>
        <p>Professional 2D Floor Plan CAD, Vastu Analysis, BOQ Cost Estimator, and Interactive 3D Walkthrough Studio — exclusively on Android.</p>
        <p style="margin-top:24px;">
          <a href="/privacy-policy">Privacy Policy</a> |
          <a href="/delete-account">Delete Account</a>
        </p>
      </div>
    </body>
    </html>
    """


@app.get("/privacy-policy", response_class=HTMLResponse)
def privacy_policy():
    return """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /><title>Privacy Policy — Floor Metric</title></head>
    <body style="font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;line-height:1.6;color:#1e293b;">
      <h1>Privacy Policy — Floor Metric</h1>
      <p><strong>Effective Date:</strong> November 2026</p>
      <p>Floor Metric collects your verified email address solely to create your account, send one-time verification codes (OTPs), sync your Lifetime Pro access and AI Architect credit balance across devices, and send important service updates.</p>
      <p>Payments are processed securely through Google Play Billing. We never store your card or bank details.</p>
      <p>You can permanently delete your account and all stored email and credit records at any time from inside the app (<strong>Profile → Delete Account</strong>) or by visiting our <a href="/delete-account">Account Deletion page</a>.</p>
    </body>
    </html>
    """


@app.get("/delete-account", response_class=HTMLResponse)
def delete_account_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /><title>Delete Account — Floor Metric</title></head>
    <body style="font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;line-height:1.6;color:#1e293b;">
      <h1>Delete Your Floor Metric Account</h1>
      <p>To immediately and permanently delete your email address, session data, and credit wallet from our database:</p>
      <ol>
        <li>Open the <strong>Floor Metric</strong> app on Android.</li>
        <li>Tap the <strong>Profile</strong> icon in the top bar.</li>
        <li>Tap <strong>Delete Account</strong> and confirm.</li>
      </ol>
      <p>All account records are deleted immediately upon confirmation. Note that deleting your account permanently removes any remaining credits and Pro status tied to that email.</p>
    </body>
    </html>
    """


# ── Email OTP Authentication Endpoints ───────────────────────────
@app.api_route("/ping", methods=["GET", "HEAD"])
def ping_server():
    return {"status": "ok"}


@app.post("/auth/send-otp")
def send_otp(req: SendOtpRequest):
    email = _normalize_email(req.email)
    now_utc = datetime.now(timezone.utc)

    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT sent_at FROM email_otps WHERE email = %s", (email,))
            row = cur.fetchone()
            if row:
                sent_at = row[0]
                if sent_at.tzinfo is None:
                    sent_at = sent_at.replace(tzinfo=timezone.utc)
                elapsed = (now_utc - sent_at).total_seconds()
                if elapsed < 30:
                    wait_sec = int(30 - elapsed) + 1
                    raise HTTPException(
                        status_code=429,
                        detail=f"Please wait {wait_sec}s before requesting another code.",
                    )

            otp = f"{secrets.randbelow(1000000):06d}"
            otp_hash = _hash_otp(email, otp)
            expires_at = now_utc + timedelta(minutes=OTP_EXPIRY_MINUTES)

            cur.execute(
                """
                INSERT INTO email_otps (email, otp_hash, expires_at, attempts, sent_at)
                VALUES (%s, %s, %s, 0, %s)
                ON CONFLICT (email) DO UPDATE
                SET otp_hash = EXCLUDED.otp_hash,
                    expires_at = EXCLUDED.expires_at,
                    attempts = 0,
                    sent_at = EXCLUDED.sent_at
                """,
                (email, otp_hash, expires_at, now_utc),
            )

    import threading
    def _bg_mail(to_e, code):
        try:
            _send_otp_email(to_e, code)
        except Exception as exc:
            print(f"[OTP_BG_ERROR] {exc}", flush=True)
    threading.Thread(target=_bg_mail, args=(email, otp), daemon=True).start()
    return {"ok": True, "message": "6-digit verification code sent to your email."}


@app.post("/auth/verify-otp")
def verify_otp(req: VerifyOtpRequest):
    email = _normalize_email(req.email)
    otp = req.otp.strip()
    if len(otp) != 6 or not otp.isdigit():
        raise HTTPException(status_code=400, detail="Enter a valid 6-digit code.")

    now_utc = datetime.now(timezone.utc)
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT otp_hash, expires_at, attempts FROM email_otps WHERE email = %s",
                (email,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="No verification code found. Request a new code.")

            stored_hash, expires_at, attempts = row
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            if now_utc > expires_at:
                cur.execute("DELETE FROM email_otps WHERE email = %s", (email,))
                raise HTTPException(status_code=400, detail="Code expired. Please request a new code.")

            if attempts >= 5:
                cur.execute("DELETE FROM email_otps WHERE email = %s", (email,))
                raise HTTPException(status_code=429, detail="Too many wrong attempts. Request a new code.")

            expected_hash = _hash_otp(email, otp)
            if not hmac.compare_digest(stored_hash, expected_hash):
                cur.execute(
                    "UPDATE email_otps SET attempts = attempts + 1 WHERE email = %s",
                    (email,),
                )
                raise HTTPException(status_code=400, detail="Incorrect 6-digit code.")

            cur.execute("DELETE FROM email_otps WHERE email = %s", (email,))

        wallet = get_and_sync_wallet(conn, email)

    token = _create_jwt(email)
    return {"token": token, "email": email, "wallet": wallet}


@app.get("/auth/me")
def get_me(email: str = Depends(get_current_email)):
    with get_db() as conn:
        wallet = get_and_sync_wallet(conn, email)
    return {"email": email, "wallet": wallet}


@app.delete("/auth/delete-account")
def delete_account(email: str = Depends(get_current_email)):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM email_otps WHERE email = %s", (email,))
            cur.execute("DELETE FROM users WHERE email = %s", (email,))
    return {"ok": True, "message": "Account permanently deleted."}


# ── Protected Pro, Credits & AI Architect Endpoints ──────────────
@app.post("/activate-pro")
def activate_pro(
    req: VerifyPurchaseRequest,
    email: str = Depends(get_current_email),
):
    if not req.purchase_token or len(req.purchase_token) < 10:
        raise HTTPException(status_code=400, detail="Invalid Google Play purchase token")
    now_utc = datetime.now(timezone.utc)
    with get_db() as conn:
        get_and_sync_wallet(conn, email)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET has_lifetime_pass = TRUE, daily_credits = 100, last_reset_utc = %s, last_restore_utc = %s
                WHERE email = %s
                """,
                (now_utc, now_utc, email),
            )
        return get_and_sync_wallet(conn, email)


@app.post("/restore-credits")
def restore_credits(
    req: VerifyPurchaseRequest,
    email: str = Depends(get_current_email),
):
    if not req.purchase_token or len(req.purchase_token) < 10:
        raise HTTPException(status_code=400, detail="Invalid Google Play purchase token")
    now_utc = datetime.now(timezone.utc)
    with get_db() as conn:
        wallet = get_and_sync_wallet(conn, email)
        if not wallet["hasLifetimePass"]:
            raise HTTPException(status_code=403, detail="Lifetime Pro Pass required.")
        if wallet.get("lastRestoreUtc"):
            last_rest = datetime.fromisoformat(wallet["lastRestoreUtc"])
            if last_rest.tzinfo is None:
                last_rest = last_rest.replace(tzinfo=timezone.utc)
            elapsed = (now_utc - last_rest).total_seconds()
            if elapsed < 86400:
                rem = int(86400 - elapsed)
                raise HTTPException(status_code=400, detail=f"Restore available in {rem // 3600:02d}:{(rem % 3600) // 60:02d}:{rem % 60:02d}.")
        restore_amt = wallet["yesterdayUnused"] if wallet["yesterdayUnused"] > 0 else 100
        new_bonus = min(100, wallet["restoredBonusCredits"] + min(100, restore_amt))
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET yesterday_unused = 0, restored_bonus_credits = %s, last_restore_utc = %s
                WHERE email = %s
                """,
                (new_bonus, now_utc, email),
            )
        return get_and_sync_wallet(conn, email)


@app.post("/generate-plan")
def generate_plan(
    req: GenerateRequest,
    email: str = Depends(get_current_email),
):
    with get_db() as conn:
        wallet = get_and_sync_wallet(conn, email)
        if not wallet["hasLifetimePass"]:
            raise HTTPException(status_code=403, detail="Lifetime Pro Pass required.")
        total_credits = wallet["dailyCredits"] + wallet["restoredBonusCredits"]
        if total_credits < 10:
            raise HTTPException(status_code=402, detail="Insufficient credits. Restore credits (₹499).")
        import json, re
        system_instruction = (
            "You are an expert Principal Architect and CAD Spatial Planner. Design a complete, intelligent, custom 2D floor plan "
            "tailored specifically to the user's exact prompt - residential (1BHK, 2BHK, 3BHK, 4BHK, villa, duplex, studio), Vastu home, "
            "commercial office, clinic, cafe, restaurant, retail shop, or custom plot dimensions (1m=100cm, 1ft=30.48cm).\n"
            "RULES: 1) +X is East (Right), +Y is South (Down), (0,0) is North-West (Top-Left). "
            "2) ZERO OVERLAPS: Stack rooms flush edge-to-edge sharing exact wall coordinates so no two rooms overlap. "
            "3) VASTU (when requested): North-East = high X, low Y (Living/Pooja); South-East = high X, high Y (Kitchen); "
            "South-West = low X, high Y (Master Bedroom); North-West = low X, low Y (Guest/Bath). "
            "4) Add connecting doors (doorLeft/doorRight, 90cm) on shared walls and windows (window, 120-180cm) on outer walls. "
            "5) Place 2-4 realistic fixtures per room using symbolType in {'bed','sofa','kitchen','bath','wardrobe','desk'} with clear labels "
            "strictly inside [25, widthCm-fw-25] x [25, lengthCm-fl-25]. Output ONLY valid JSON matching HousePlanSchema."
        )
        def _ov(a, b, tol=1.0):
            return a["offsetXCm"] < b["offsetXCm"] + b["widthCm"] - tol and a["offsetXCm"] + a["widthCm"] > b["offsetXCm"] + tol and a["offsetYCm"] < b["offsetYCm"] + b["lengthCm"] - tol and a["offsetYCm"] + a["lengthCm"] > b["offsetYCm"] + tol
        def _sw(a, b, tol=5.0):
            if abs((a["offsetXCm"] + a["widthCm"]) - b["offsetXCm"]) <= tol and min(a["offsetYCm"] + a["lengthCm"], b["offsetYCm"] + b["lengthCm"]) - max(a["offsetYCm"], b["offsetYCm"]) >= 80: return "east"
            if abs(a["offsetXCm"] - (b["offsetXCm"] + b["widthCm"])) <= tol and min(a["offsetYCm"] + a["lengthCm"], b["offsetYCm"] + b["lengthCm"]) - max(a["offsetYCm"], b["offsetYCm"]) >= 80: return "west"
            if abs((a["offsetYCm"] + a["lengthCm"]) - b["offsetYCm"]) <= tol and min(a["offsetXCm"] + a["widthCm"], b["offsetXCm"] + b["widthCm"]) - max(a["offsetXCm"], b["offsetXCm"]) >= 80: return "south"
            if abs(a["offsetYCm"] - (b["offsetYCm"] + b["lengthCm"])) <= tol and min(a["offsetXCm"] + a["widthCm"], b["offsetXCm"] + b["widthCm"]) - max(a["offsetXCm"], b["offsetXCm"]) >= 80: return "north"
            return None
        def clean_plan(raw_text: str) -> str:
            txt = raw_text.strip()
            if txt.startswith("```"): txt = re.sub(r"```(?:json)?\s*", "", txt); txt = re.sub(r"\s*```$", "", txt)
            o = json.loads(txt)
            if isinstance(o, dict) and "rooms" not in o:
                for v in o.values():
                    if isinstance(v, dict) and "rooms" in v: o = v; break
            o["id"] = str(o.get("id") or f"plan_{int(datetime.now(timezone.utc).timestamp())}")
            o["projectName"] = str(o.get("projectName") or req.prompt[:42].upper())
            o["architectName"] = "FLOOR METRIC AI STUDIO"
            raw_rooms = o.get("rooms") or []
            if not isinstance(raw_rooms, list) or not raw_rooms: raise ValueError("No rooms")
            cleaned = []
            for ri, r in enumerate(raw_rooms):
                if not isinstance(r, dict): continue
                rn = str(r.get("name") or f"Room {ri+1}").strip()
                rw = round(max(140.0, min(1800.0, float(r.get("widthCm", 360.0)))) / 10.0) * 10.0
                rl = round(max(140.0, min(1800.0, float(r.get("lengthCm", 330.0)))) / 10.0) * 10.0
                ox = round(float(r.get("offsetXCm", 0.0)) / 10.0) * 10.0
                oy = round(float(r.get("offsetYCm", 0.0)) / 10.0) * 10.0
                wt = float(r.get("wallThicknessCm", 23.0))
                if wt not in (11.5, 23.0): wt = 23.0 if rw >= 330.0 or rl >= 330.0 else 11.5
                ops = []
                for oi, op in enumerate(r.get("openings") or []):
                    if not isinstance(op, dict): continue
                    ot = str(op.get("type", "doorLeft")).lower()
                    opt = "window" if "win" in ot else ("doorRight" if "right" in ot else "doorLeft")
                    wl = str(op.get("wall", "south")).lower()
                    wall = "north" if wl.startswith(("n","top","up")) else ("east" if wl.startswith(("e","right")) else ("west" if wl.startswith(("w","left")) else "south"))
                    wlen = rw if wall in ("north","south") else rl
                    ow = min(float(op.get("widthCm", 120.0 if opt=="window" else 90.0)), max(60.0, wlen - 40.0))
                    ops.append({"id": str(op.get("id") or f"op_{ri+1}_{oi+1}"), "type": opt, "wall": wall, "offsetCm": max(20.0, min(float(op.get("offsetCm", 45.0)), max(20.0, wlen - ow - 20.0))), "widthCm": ow})
                fxs = []
                for fi, fx in enumerate(r.get("fixtures") or []):
                    if not isinstance(fx, dict): continue
                    lb = str(fx.get("label") or "Fixture").strip()
                    st = (str(fx.get("symbolType", "")) + " " + lb).lower()
                    sym = "bed" if any(k in st for k in ("bed","cot","mattress")) else ("sofa" if any(k in st for k in ("sofa","couch","lounge","reception","waiting")) else ("kitchen" if any(k in st for k in ("kitchen","stove","cook","pantry","counter","island")) else ("bath" if any(k in st for k in ("bath","toilet","wc","shower","sink","basin","wash")) else ("wardrobe" if any(k in st for k in ("wardrobe","closet","cupboard","almirah","cabinet")) else "desk"))))
                    fw = min(max(45.0, float(fx.get("widthCm", 110.0))), max(45.0, rw - 40.0))
                    fl = min(max(45.0, float(fx.get("lengthCm", 75.0))), max(45.0, rl - 40.0))
                    fxs.append({"id": str(fx.get("id") or f"fix_{ri+1}_{fi+1}"), "label": lb, "symbolType": sym, "xCm": max(20.0, min(float(fx.get("xCm", (rw-fw)/2)), rw-fw-20.0)), "yCm": max(20.0, min(float(fx.get("yCm", (rl-fl)/2)), rl-fl-20.0)), "widthCm": fw, "lengthCm": fl, "rotationDeg": float(fx.get("rotationDeg", 0.0)) % 360.0})
                if not fxs:
                    nl = rn.lower()
                    if any(k in nl for k in ("bed","master","suite","guest")): fxs.append({"id": f"fix_{ri+1}_1", "label": "Queen Bed", "symbolType": "bed", "xCm": 25.0, "yCm": 25.0, "widthCm": min(160.0, rw-45.0), "lengthCm": min(190.0, rl-45.0), "rotationDeg": 0.0})
                    elif any(k in nl for k in ("living","hall","lounge","foyer")): fxs.append({"id": f"fix_{ri+1}_1", "label": "3-Seat Sofa", "symbolType": "sofa", "xCm": 30.0, "yCm": 30.0, "widthCm": min(200.0, rw-50.0), "lengthCm": min(85.0, rl-50.0), "rotationDeg": 0.0})
                    elif any(k in nl for k in ("kitchen","pantry","cafe")): fxs.append({"id": f"fix_{ri+1}_1", "label": "Kitchen Counter", "symbolType": "kitchen", "xCm": 25.0, "yCm": 25.0, "widthCm": min(210.0, rw-45.0), "lengthCm": min(65.0, rl-45.0), "rotationDeg": 0.0})
                    elif any(k in nl for k in ("bath","wc","toilet","wash","restroom")): fxs.append({"id": f"fix_{ri+1}_1", "label": "Bath Suite", "symbolType": "bath", "xCm": 20.0, "yCm": 20.0, "widthCm": min(150.0, rw-35.0), "lengthCm": min(75.0, rl-35.0), "rotationDeg": 0.0})
                    elif any(k in nl for k in ("office","study","cabin","work","clinic","desk")): fxs.append({"id": f"fix_{ri+1}_1", "label": "Work Desk", "symbolType": "desk", "xCm": 30.0, "yCm": 30.0, "widthCm": min(140.0, rw-50.0), "lengthCm": min(70.0, rl-50.0), "rotationDeg": 0.0})
                cleaned.append({"id": str(r.get("id") or f"room_{ri+1}"), "name": rn, "offsetXCm": ox, "offsetYCm": oy, "widthCm": rw, "lengthCm": rl, "wallThicknessCm": wt, "floorLevel": int(req.floor_level), "openings": ops, "fixtures": fxs})
            placed = []
            for rm in cleaned:
                if not placed:
                    rm["offsetXCm"], rm["offsetYCm"] = 0.0, 0.0; placed.append(rm); continue
                if any(_ov(rm, p) for p in placed):
                    cands = []
                    for p in placed:
                        cands.extend([(p["offsetXCm"]+p["widthCm"], p["offsetYCm"]), (p["offsetXCm"], p["offsetYCm"]+p["lengthCm"]), (p["offsetXCm"]-rm["widthCm"], p["offsetYCm"]), (p["offsetXCm"], p["offsetYCm"]-rm["lengthCm"])])
                    valid = [(abs(cx-rm["offsetXCm"])+abs(cy-rm["offsetYCm"]), cx, cy) for cx, cy in cands if not any(_ov(dict(rm, offsetXCm=cx, offsetYCm=cy), p) for p in placed)]
                    if valid:
                        valid.sort(key=lambda t: t[0]); rm["offsetXCm"], rm["offsetYCm"] = valid[0][1], valid[0][2]
                placed.append(rm)
            mx, my = min(r["offsetXCm"] for r in placed), min(r["offsetYCm"] for r in placed)
            for r in placed: r["offsetXCm"] -= mx; r["offsetYCm"] -= my
            for idx, r in enumerate(placed):
                if not any(op["type"] in ("doorLeft","doorRight") for op in r["openings"]):
                    dw = next((_sw(r, o) for k, o in enumerate(placed) if k != idx and _sw(r, o)), "south")
                    wl = r["widthCm"] if dw in ("north","south") else r["lengthCm"]
                    r["openings"].append({"id": f"ad_{idx+1}", "type": "doorLeft", "wall": dw, "offsetCm": max(20.0, min(40.0, wl-110.0)), "widthCm": min(90.0, max(65.0, wl-40.0))})
                if not any(op["type"] == "window" for op in r["openings"]):
                    occ = {_sw(r, o) for k, o in enumerate(placed) if k != idx}
                    for cw in ("north","east","west","south"):
                        if cw not in occ:
                            wl = r["widthCm"] if cw in ("north","south") else r["lengthCm"]
                            if wl >= 120.0:
                                r["openings"].append({"id": f"aw_{idx+1}", "type": "window", "wall": cw, "offsetCm": max(25.0, (wl-min(140.0, wl-50.0))/2), "widthCm": min(140.0, wl-50.0)}); break
            o["rooms"] = placed
            return HousePlanSchema.model_validate(o).model_dump_json()
        plan_json_str = None
        user_prompt = f"Target Floor Level: {req.floor_level}.\nUser Request: {req.prompt}\nDesign all requested rooms with accurate dimensions, shared flush walls, doors, windows, and rich furniture."
        g_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if g_key:
            gc = genai.Client(api_key=g_key)
            for gm in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
                if plan_json_str: break
                try:
                    try:
                        cfg = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=HousePlanSchema, temperature=0.35, thinking_config=types.ThinkingConfig(thinking_budget=0))
                    except TypeError:
                        cfg = types.GenerateContentConfig(system_instruction=system_instruction, response_mime_type="application/json", response_schema=HousePlanSchema, temperature=0.35)
                    resp = gc.models.generate_content(model=gm, contents=user_prompt, config=cfg)
                    if resp and resp.text:
                        plan_json_str = clean_plan(resp.text); print(f"[AI_SUCCESS] Gemini:{gm}", flush=True); break
                except Exception as e:
                    print(f"[AI_FALLBACK] Gemini:{gm} failed: {e}", flush=True)
        hint = " JSON keys: {id,projectName,architectName,rooms:[{id,name,offsetXCm,offsetYCm,widthCm,lengthCm,wallThicknessCm,floorLevel,openings:[{id,type(doorLeft/doorRight/window),wall(north/south/east/west),offsetCm,widthCm}],fixtures:[{id,label,symbolType(bed/sofa/kitchen/bath/desk/wardrobe),xCm,yCm,widthCm,lengthCm,rotationDeg}]}]}"
        providers = []
        gk = os.environ.get("GROQ_API_KEY", "").strip()
        if gk:
            for gm in ("llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant"):
                providers.append(("Groq:" + gm, "https://" + "api." + "groq.com/openai/v1/chat/completions", gk, gm))
        ork = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if ork:
            for om in ("meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-flash-exp:free", "openrouter/free"):
                providers.append(("OpenRouter:" + om, "https://" + "openrouter." + "ai/api/v1/chat/completions", ork, om))
        mk = os.environ.get("MISTRAL_API_KEY", "").strip()
        if mk:
            providers.append(("Mistral:mistral-small-latest", "https://" + "api." + "mistral.ai/v1/chat/completions", mk, "mistral-small-latest"))
        for pname, purl, pkey, pmodel in providers:
            if plan_json_str: break
            try:
                pr_resp = requests.post(purl, headers={"Authorization": f"Bearer {pkey}", "Content-Type": "application/json"}, json={"model": pmodel, "messages": [{"role": "system", "content": system_instruction + hint}, {"role": "user", "content": user_prompt}], "response_format": {"type": "json_object"}, "temperature": 0.35, "max_tokens": 4096}, timeout=25)
                if pr_resp.status_code >= 400:
                    print(f"[AI_FALLBACK] {pname} HTTP {pr_resp.status_code}: {pr_resp.text[:160]}", flush=True); continue
                plan_json_str = clean_plan(pr_resp.json()["choices"][0]["message"]["content"])
                print(f"[AI_SUCCESS] {pname}", flush=True); break
            except Exception as ex:
                print(f"[AI_FALLBACK] {pname} error: {ex}", flush=True); continue
        if not plan_json_str:
            raise HTTPException(status_code=400, detail="AI generation failed. Please try again.")
        daily, bonus = wallet["dailyCredits"], wallet["restoredBonusCredits"]
        if daily >= 10: daily -= 10
        else: rem = 10 - daily; daily = 0; bonus -= rem
        with conn.cursor() as cur:
            cur.execute("UPDATE users SET daily_credits = %s, restored_bonus_credits = %s WHERE email = %s", (daily, bonus, email))
        updated_wallet = get_and_sync_wallet(conn, email)
        return {"wallet": updated_wallet, "plan_json": plan_json_str}

