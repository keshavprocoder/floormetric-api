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


@app.post('/generate-plan')
def generate_plan(
    req: GenerateRequest,
    email: str = Depends(get_current_email),
):
    with get_db() as conn:
        wallet = get_and_sync_wallet(conn, email)
        if not wallet['hasLifetimePass']:
            raise HTTPException(status_code=403, detail='Lifetime Pro Pass required.')
        total_credits = wallet['dailyCredits'] + wallet['restoredBonusCredits']
        if total_credits < 10:
            raise HTTPException(status_code=402, detail='Insufficient credits. Restore credits (₹499).')
        import json, re
        p_txt = (req.prompt or '').strip()
        if len(p_txt) < 3 or not re.search(r'[A-Za-z]', p_txt):
            raise HTTPException(status_code=400, detail='Please describe a floor plan or room layout (e.g. 2BHK Vastu home or Dental clinic).')
        system_instruction = (
            'You are an expert Principal Architect and CAD Spatial Planner. Design a complete, realistic, walkable 2D/3D floor plan '
            'strictly fulfilling the exact user request (any residential 1BHK-6BHK, villa, duplex, Vastu home, office, clinic, cafe, shop, gym, or custom plot; 1m=100cm, 1ft=30.48cm).\n'
            'RULES: 1) +X is East (Right), +Y is South (Down), (0,0) is North-West (Top-Left). '
            '2) ZERO OVERLAPS AND ZERO GAPS: Every room MUST touch adjacent rooms flush edge-to-edge sharing >=110cm wall length. '
            '3) FULL CONNECTIVITY: Every room MUST be interconnected via aligned 90cm doors on shared walls (matching doors on both touching rooms) plus ONE main exterior entrance door. '
            '4) EXTERIOR WINDOWS ONLY: Place windows ONLY on outer walls facing outside; NEVER place windows on shared interior walls or overlapping any door. '
            '5) CLEAR DOORWAYS: Keep >=110cm clearance in front of every door completely free of furniture; place 1-3 realistic fixtures inside room bounds using symbolType in {bed,sofa,kitchen,bath,wardrobe,desk}. '
            '6) VASTU (when requested): NE = high X, low Y (Living/Pooja); SE = high X, high Y (Kitchen); SW = low X, high Y (Master Bed); NW = low X, low Y (Guest/Bath). '
            '7) If the request is NOT for designing a floor plan, building, room, plot, home, office, shop, clinic, or architectural space (e.g. jokes, coding, math, general chat), output ONLY {"rejected": true, "reason": "Please enter a floor plan or building layout request."}. '
            'Output ONLY valid JSON matching HousePlanSchema.'
        )
        def _ov(a, b, tol=1.0):
            return a['offsetXCm'] < b['offsetXCm'] + b['widthCm'] - tol and a['offsetXCm'] + a['widthCm'] > b['offsetXCm'] + tol and a['offsetYCm'] < b['offsetYCm'] + b['lengthCm'] - tol and a['offsetYCm'] + a['lengthCm'] > b['offsetYCm'] + tol
        def _shared_seg(a, b, min_len=100.0, tol=5.0):
            if abs((a['offsetXCm'] + a['widthCm']) - b['offsetXCm']) <= tol:
                s0, s1 = max(a['offsetYCm'], b['offsetYCm']), min(a['offsetYCm'] + a['lengthCm'], b['offsetYCm'] + b['lengthCm'])
                if s1 - s0 >= min_len: return ('east', 'west', s0, s1)
            if abs(a['offsetXCm'] - (b['offsetXCm'] + b['widthCm'])) <= tol:
                s0, s1 = max(a['offsetYCm'], b['offsetYCm']), min(a['offsetYCm'] + a['lengthCm'], b['offsetYCm'] + b['lengthCm'])
                if s1 - s0 >= min_len: return ('west', 'east', s0, s1)
            if abs((a['offsetYCm'] + a['lengthCm']) - b['offsetYCm']) <= tol:
                s0, s1 = max(a['offsetXCm'], b['offsetXCm']), min(a['offsetXCm'] + a['widthCm'], b['offsetXCm'] + b['widthCm'])
                if s1 - s0 >= min_len: return ('south', 'north', s0, s1)
            if abs(a['offsetYCm'] - (b['offsetYCm'] + b['lengthCm'])) <= tol:
                s0, s1 = max(a['offsetXCm'], b['offsetXCm']), min(a['offsetXCm'] + a['widthCm'], b['offsetXCm'] + b['widthCm'])
                if s1 - s0 >= min_len: return ('north', 'south', s0, s1)
            return None
        def _worig(r, w):
            return r['offsetXCm'] if w in ('north', 'south') else r['offsetYCm']
        def _wlen(r, w):
            return r['widthCm'] if w in ('north', 'south') else r['lengthCm']
        def _iov(a0, a1, b0, b1, pad=25.0):
            return a0 < b1 + pad and b0 < a1 + pad
        def _rov(a, b, pad=0.0):
            return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad
        def clean_plan(raw_text) -> str:
            if not raw_text or not isinstance(raw_text, str): raise ValueError('Empty AI output')
            txt = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.S).strip()
            st_i, en_i = txt.find('{'), txt.rfind('}')
            if st_i == -1 or en_i <= st_i: raise ValueError('No JSON object found')
            o = json.loads(txt[st_i:en_i+1])
            if isinstance(o, dict) and o.get('rejected'):
                raise HTTPException(status_code=400, detail=str(o.get('reason') or 'Please enter a valid floor plan or building layout request.'))
            if isinstance(o, dict) and 'rooms' not in o:
                for v in o.values():
                    if isinstance(v, dict) and 'rooms' in v: o = v; break
            o['id'] = str(o.get('id') or f'plan_{int(datetime.now(timezone.utc).timestamp())}')
            o['projectName'] = str(o.get('projectName') or req.prompt[:42].upper())
            o['architectName'] = 'FLOOR METRIC STUDIO'
            raw_rooms = o.get('rooms') or []
            if not isinstance(raw_rooms, list) or not raw_rooms: raise ValueError('No rooms')
            cleaned = []
            for ri, r in enumerate(raw_rooms):
                if not isinstance(r, dict): continue
                rn = str(r.get('name') or f'Room {ri+1}').strip()
                rw = round(max(160.0, min(1800.0, float(r.get('widthCm', 360.0)))) / 10.0) * 10.0
                rl = round(max(160.0, min(1800.0, float(r.get('lengthCm', 330.0)))) / 10.0) * 10.0
                ox = round(float(r.get('offsetXCm', 0.0)) / 10.0) * 10.0
                oy = round(float(r.get('offsetYCm', 0.0)) / 10.0) * 10.0
                wt = float(r.get('wallThicknessCm', 23.0))
                if wt not in (11.5, 23.0): wt = 23.0 if rw >= 330.0 or rl >= 330.0 else 11.5
                ops = []
                for oi, op in enumerate(r.get('openings') or []):
                    if not isinstance(op, dict): continue
                    ot = str(op.get('type', 'doorLeft')).lower()
                    opt = 'window' if 'win' in ot else ('doorRight' if 'right' in ot else 'doorLeft')
                    wl = str(op.get('wall', 'south')).lower()
                    wall = 'north' if wl.startswith(('n','top','up')) else ('east' if wl.startswith(('e','right')) else ('west' if wl.startswith(('w','left')) else 'south'))
                    wlen = rw if wall in ('north','south') else rl
                    ow = min(float(op.get('widthCm', 120.0 if opt=='window' else 90.0)), max(65.0, wlen - 30.0))
                    ops.append({'id': str(op.get('id') or f'op_{ri+1}_{oi+1}'), 'type': opt, 'wall': wall, 'offsetCm': max(15.0, min(float(op.get('offsetCm', 45.0)), max(15.0, wlen - ow - 15.0))), 'widthCm': ow})
                fxs = []
                for fi, fx in enumerate(r.get('fixtures') or []):
                    if not isinstance(fx, dict): continue
                    lb = str(fx.get('label') or 'Fixture').strip()
                    st = (str(fx.get('symbolType', '')) + ' ' + lb).lower()
                    sym = 'bed' if any(k in st for k in ('bed','cot','mattress')) else ('sofa' if any(k in st for k in ('sofa','couch','lounge','reception','waiting')) else ('kitchen' if any(k in st for k in ('kitchen','stove','cook','pantry','counter','island')) else ('bath' if any(k in st for k in ('bath','toilet','wc','shower','sink','basin','wash')) else ('wardrobe' if any(k in st for k in ('wardrobe','closet','cupboard','almirah','cabinet')) else 'desk'))))
                    fw = min(max(45.0, float(fx.get('widthCm', 110.0))), max(45.0, rw - 36.0))
                    fl = min(max(45.0, float(fx.get('lengthCm', 75.0))), max(45.0, rl - 36.0))
                    fxs.append({'id': str(fx.get('id') or f'fix_{ri+1}_{fi+1}'), 'label': lb, 'symbolType': sym, 'xCm': max(16.0, min(float(fx.get('xCm', (rw-fw)/2)), rw-fw-16.0)), 'yCm': max(16.0, min(float(fx.get('yCm', (rl-fl)/2)), rl-fl-16.0)), 'widthCm': fw, 'lengthCm': fl, 'rotationDeg': float(fx.get('rotationDeg', 0.0))})
                if not fxs:
                    nl = rn.lower()
                    if any(k in nl for k in ('bed','master','suite','guest')): fxs.append({'id': f'fix_{ri+1}_1', 'label': 'Queen Bed', 'symbolType': 'bed', 'xCm': 20.0, 'yCm': 20.0, 'widthCm': min(160.0, rw-40.0), 'lengthCm': min(190.0, rl-40.0), 'rotationDeg': 0.0})
                    elif any(k in nl for k in ('living','hall','lounge','foyer')): fxs.append({'id': f'fix_{ri+1}_1', 'label': '3-Seat Sofa', 'symbolType': 'sofa', 'xCm': 25.0, 'yCm': 25.0, 'widthCm': min(190.0, rw-45.0), 'lengthCm': min(85.0, rl-45.0), 'rotationDeg': 0.0})
                    elif any(k in nl for k in ('kitchen','pantry','cafe')): fxs.append({'id': f'fix_{ri+1}_1', 'label': 'Kitchen Counter', 'symbolType': 'kitchen', 'xCm': 20.0, 'yCm': 20.0, 'widthCm': min(200.0, rw-40.0), 'lengthCm': min(65.0, rl-40.0), 'rotationDeg': 0.0})
                    elif any(k in nl for k in ('bath','wc','toilet','wash','restroom')): fxs.append({'id': f'fix_{ri+1}_1', 'label': 'Bath Suite', 'symbolType': 'bath', 'xCm': 18.0, 'yCm': 18.0, 'widthCm': min(130.0, rw-36.0), 'lengthCm': min(70.0, rl-36.0), 'rotationDeg': 0.0})
                    elif any(k in nl for k in ('office','study','cabin','work','clinic','desk')): fxs.append({'id': f'fix_{ri+1}_1', 'label': 'Work Desk', 'symbolType': 'desk', 'xCm': 25.0, 'yCm': 25.0, 'widthCm': min(135.0, rw-45.0), 'lengthCm': min(68.0, rl-45.0), 'rotationDeg': 0.0})
                cleaned.append({'id': str(r.get('id') or f'room_{ri+1}'), 'name': rn, 'offsetXCm': ox, 'offsetYCm': oy, 'widthCm': rw, 'lengthCm': rl, 'wallThicknessCm': wt, 'floorLevel': int(req.floor_level), 'openings': ops, 'fixtures': fxs})
            placed = []
            for rm in cleaned:
                if not placed:
                    placed.append(rm); continue
                needs_snap = any(_ov(rm, p) for p in placed) or not any(_shared_seg(rm, p, 100.0) for p in placed)
                if needs_snap:
                    cands = []
                    for p in placed:
                        cands.extend([(p['offsetXCm']+p['widthCm'], p['offsetYCm']), (p['offsetXCm']+p['widthCm'], p['offsetYCm']+p['lengthCm']-rm['lengthCm']), (p['offsetXCm']+p['widthCm'], p['offsetYCm']+(p['lengthCm']-rm['lengthCm'])/2.0), (p['offsetXCm']-rm['widthCm'], p['offsetYCm']), (p['offsetXCm']-rm['widthCm'], p['offsetYCm']+p['lengthCm']-rm['lengthCm']), (p['offsetXCm']-rm['widthCm'], p['offsetYCm']+(p['lengthCm']-rm['lengthCm'])/2.0), (p['offsetXCm'], p['offsetYCm']+p['lengthCm']), (p['offsetXCm']+p['widthCm']-rm['widthCm'], p['offsetYCm']+p['lengthCm']), (p['offsetXCm']+(p['widthCm']-rm['widthCm'])/2.0, p['offsetYCm']+p['lengthCm']), (p['offsetXCm'], p['offsetYCm']-rm['lengthCm']), (p['offsetXCm']+p['widthCm']-rm['widthCm'], p['offsetYCm']-rm['lengthCm']), (p['offsetXCm']+(p['widthCm']-rm['widthCm'])/2.0, p['offsetYCm']-rm['lengthCm'])])
                    valid = []
                    for cx, cy in cands:
                        cand = dict(rm, offsetXCm=cx, offsetYCm=cy)
                        if not any(_ov(cand, p) for p in placed) and any(_shared_seg(cand, p, 100.0) for p in placed):
                            valid.append((abs(cx-rm['offsetXCm'])+abs(cy-rm['offsetYCm']), cx, cy))
                    if valid:
                        valid.sort(key=lambda t: t[0]); rm['offsetXCm'], rm['offsetYCm'] = valid[0][1], valid[0][2]
                placed.append(rm)
            mx, my = min(r['offsetXCm'] for r in placed), min(r['offsetYCm'] for r in placed)
            for r in placed: r['offsetXCm'] = round(r['offsetXCm'] - mx, 1); r['offsetYCm'] = round(r['offsetYCm'] - my, 1)
            n = len(placed)
            edges = []
            shared_spans = [{w: [] for w in ('north','south','east','west')} for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    sh_any = _shared_seg(placed[i], placed[j], 10.0)
                    if sh_any:
                        wi, wj, s0, s1 = sh_any
                        shared_spans[i][wi].append((s0, s1))
                        shared_spans[j][wj].append((s0, s1))
                        if s1 - s0 >= 100.0:
                            edges.append((i, j, wi, wj, s0, s1))
            par = list(range(n))
            def _find(x):
                while par[x] != x:
                    par[x] = par[par[x]]; x = par[x]
                return x
            new_doors = [[] for _ in range(n)]
            new_wins = [[] for _ in range(n)]
            conn_pairs = set()
            d_seq = [0]
            def _add_door_pair(i, j, wi, wj, s0, s1, pref_ws=None):
                lo, hi = s0 + 12.0, s1 - 12.0
                dw = min(90.0, hi - lo)
                if dw < 75.0:
                    lo, hi = s0 + 5.0, s1 - 5.0; dw = min(90.0, hi - lo)
                if dw < 65.0: return False
                center_ws = (lo + hi - dw) / 2.0
                c_ws = []
                if pref_ws is not None: c_ws.append(max(lo, min(pref_ws, hi - dw)))
                c_ws.extend([center_ws, lo, hi - dw])
                oi_base, oj_base = _worig(placed[i], wi), _worig(placed[j], wj)
                for ws in c_ws:
                    off_i, off_j = round(ws - oi_base, 1), round(ws - oj_base, 1)
                    if any(_iov(off_i, off_i + dw, d['offsetCm'], d['offsetCm'] + d['widthCm'], 25.0) for d in new_doors[i] if d['wall'] == wi): continue
                    if any(_iov(off_j, off_j + dw, d['offsetCm'], d['offsetCm'] + d['widthCm'], 25.0) for d in new_doors[j] if d['wall'] == wj): continue
                    d_seq[0] += 1
                    new_doors[i].append({'id': f'dr_{i+1}_{d_seq[0]}', 'type': 'doorLeft', 'wall': wi, 'offsetCm': off_i, 'widthCm': round(dw, 1)})
                    new_doors[j].append({'id': f'dr_{j+1}_{d_seq[0]}', 'type': 'doorRight', 'wall': wj, 'offsetCm': off_j, 'widthCm': round(dw, 1)})
                    ri_p, rj_p = _find(i), _find(j)
                    if ri_p != rj_p: par[ri_p] = rj_p
                    conn_pairs.add((min(i, j), max(i, j)))
                    return True
                return False
            for i in range(n):
                for op in placed[i]['openings']:
                    if op['type'] not in ('doorLeft', 'doorRight'): continue
                    w = op['wall']
                    ws = _worig(placed[i], w) + op['offsetCm']
                    we = ws + op['widthCm']
                    for ei, ej, wi, wj, s0, s1 in edges:
                        if (ei == i and wi == w) or (ej == i and wj == w):
                            pair = (ei, ej)
                            if pair in conn_pairs: continue
                            if min(we, s1) - max(ws, s0) >= 40.0:
                                _add_door_pair(ei, ej, wi, wj, s0, s1, pref_ws=ws); break
            hub_kw = ('living','hall','corridor','passage','lobby','foyer','dining','reception','lounge','family','common','waiting')
            minor_kw = ('bath','toilet','wc','wash','restroom','balcony','store','utility','powder')
            def _edge_rank(e):
                ei, ej, _, _, s0, s1 = e
                ni, nj = placed[ei]['name'].lower(), placed[ej]['name'].lower()
                is_hub = any(k in ni for k in hub_kw) or any(k in nj for k in hub_kw)
                is_min = any(k in ni for k in minor_kw) and any(k in nj for k in minor_kw)
                return (0 if is_hub else (2 if is_min else 1), -(s1 - s0))
            for ei, ej, wi, wj, s0, s1 in sorted(edges, key=_edge_rank):
                if _find(ei) != _find(ej):
                    _add_door_pair(ei, ej, wi, wj, s0, s1)
            def _free_ext_segs(idx, w):
                o0 = _worig(placed[idx], w)
                segs = [(o0, o0 + _wlen(placed[idx], w))]
                for s0, s1 in sorted(shared_spans[idx][w]):
                    nxt = []
                    for a, b in segs:
                        if s1 <= a or s0 >= b: nxt.append((a, b))
                        else:
                            if s0 > a: nxt.append((a, s0))
                            if s1 < b: nxt.append((s1, b))
                    segs = nxt
                return [(a, b) for a, b in segs if b - a >= 90.0]
            entry_order = sorted(range(n), key=lambda idx: (0 if any(k in placed[idx]['name'].lower() for k in hub_kw) else 1, idx))
            entry_placed = False
            for idx in entry_order:
                if entry_placed: break
                best_ext = None
                for w in ('south','east','north','west'):
                    for a, b in _free_ext_segs(idx, w):
                        if b - a >= 120.0 and (best_ext is None or (b - a) > best_ext[0]):
                            best_ext = (b - a, w, a, b)
                if best_ext:
                    _, w, a, b = best_ext
                    dw = min(100.0, b - a - 30.0)
                    off = round(((a + b - dw) / 2.0) - _worig(placed[idx], w), 1)
                    if not any(_iov(off, off + dw, d['offsetCm'], d['offsetCm'] + d['widthCm'], 25.0) for d in new_doors[idx] if d['wall'] == w):
                        new_doors[idx].append({'id': f'main_entry_{idx+1}', 'type': 'doorLeft', 'wall': w, 'offsetCm': off, 'widthCm': round(dw, 1)})
                        entry_placed = True
            for i in range(n):
                w_seq = 0
                def _can_place_win(w, off, ww):
                    all_ops = [op for op in (new_doors[i] + new_wins[i]) if op['wall'] == w]
                    return not any(_iov(off, off + ww, op['offsetCm'], op['offsetCm'] + op['widthCm'], 25.0) for op in all_ops)
                for op in placed[i]['openings']:
                    if op['type'] != 'window': continue
                    w = op['wall']
                    ws = _worig(placed[i], w) + op['offsetCm']
                    ww = min(160.0, max(80.0, op['widthCm']))
                    for a, b in _free_ext_segs(i, w):
                        if b - a >= ww + 30.0 and ws >= a - 20.0 and ws <= b + 20.0:
                            cl_ws = max(a + 15.0, min(ws, b - ww - 15.0))
                            off = round(cl_ws - _worig(placed[i], w), 1)
                            if _can_place_win(w, off, ww):
                                w_seq += 1; new_wins[i].append({'id': f'win_{i+1}_{w_seq}', 'type': 'window', 'wall': w, 'offsetCm': off, 'widthCm': round(ww, 1)}); break
                if not new_wins[i]:
                    ext_cands = []
                    for w in ('north','east','south','west'):
                        for a, b in _free_ext_segs(i, w):
                            if b - a >= 100.0: ext_cands.append((b - a, w, a, b))
                    for span, w, a, b in sorted(ext_cands, reverse=True):
                        ww = min(140.0, max(75.0, span - 40.0))
                        o_base = _worig(placed[i], w)
                        for ws in ((a + b - ww) / 2.0, a + 20.0, b - ww - 20.0):
                            off = round(ws - o_base, 1)
                            if _can_place_win(w, off, ww):
                                w_seq += 1; new_wins[i].append({'id': f'win_{i+1}_{w_seq}', 'type': 'window', 'wall': w, 'offsetCm': off, 'widthCm': round(ww, 1)}); break
                        if new_wins[i]: break
                placed[i]['openings'] = new_doors[i] + new_wins[i]
            for r in placed:
                rw, rl = r['widthCm'], r['lengthCm']
                dzones = []
                for op in r['openings']:
                    if op['type'] not in ('doorLeft', 'doorRight'): continue
                    o0, o1 = op['offsetCm'] - 15.0, op['offsetCm'] + op['widthCm'] + 15.0
                    if op['wall'] == 'north': dzones.append((o0, 0.0, o1, min(105.0, rl * 0.55)))
                    elif op['wall'] == 'south': dzones.append((o0, max(0.0, rl - min(105.0, rl * 0.55)), o1, rl))
                    elif op['wall'] == 'west': dzones.append((0.0, o0, min(105.0, rw * 0.55), o1))
                    elif op['wall'] == 'east': dzones.append((max(0.0, rw - min(105.0, rw * 0.55)), o0, rw, o1))
                good_fxs = []
                good_rects = []
                for fx in r['fixtures']:
                    orig_x, orig_y = fx['xCm'], fx['yCm']
                    placed_fx = False
                    for scale in (1.0, 0.82):
                        fw = round(max(45.0, min(fx['widthCm'] * scale, rw - 34.0)), 1)
                        fl = round(max(45.0, min(fx['lengthCm'] * scale, rl - 34.0)), 1)
                        if fw + 30.0 > rw or fl + 30.0 > rl: continue
                        def _ok(x, y):
                            if x < 15.0 or y < 15.0 or x + fw > rw - 15.0 or y + fl > rl - 15.0: return False
                            rc = (x, y, x + fw, y + fl)
                            if any(_rov(rc, dz, 0.0) for dz in dzones): return False
                            if any(_rov(rc, gr, 12.0) for gr in good_rects): return False
                            return True
                        cands = [(orig_x, orig_y), (16.0, 16.0), (rw - fw - 16.0, 16.0), (16.0, rl - fl - 16.0), (rw - fw - 16.0, rl - fl - 16.0), ((rw - fw) / 2.0, 16.0), ((rw - fw) / 2.0, rl - fl - 16.0), (16.0, (rl - fl) / 2.0), (rw - fw - 16.0, (rl - fl) / 2.0), ((rw - fw) / 2.0, (rl - fl) / 2.0)]
                        gx = 16.0
                        while gx <= rw - fw - 16.0:
                            gy = 16.0
                            while gy <= rl - fl - 16.0:
                                cands.append((gx, gy)); gy += 40.0
                            gx += 40.0
                        valid_c = [(abs(cx - orig_x) + abs(cy - orig_y), cx, cy) for cx, cy in cands if _ok(cx, cy)]
                        if valid_c:
                            valid_c.sort(key=lambda t: t[0])
                            bx, by = round(valid_c[0][1], 1), round(valid_c[0][2], 1)
                            fx['xCm'], fx['yCm'], fx['widthCm'], fx['lengthCm'] = bx, by, fw, fl
                            good_fxs.append(fx); good_rects.append((bx, by, bx + fw, by + fl)); placed_fx = True; break
                    if not placed_fx: continue
                r['fixtures'] = good_fxs
            o['rooms'] = placed
            return HousePlanSchema.model_validate(o).model_dump_json()
        plan_json_str = None
        hint = ' JSON schema: {id,projectName,architectName,rooms:[{id,name,offsetXCm,offsetYCm,widthCm,lengthCm,wallThicknessCm,floorLevel,openings:[{id,type(doorLeft/doorRight/window),wall(north/south/east/west),offsetCm,widthCm}],fixtures:[{id,label,symbolType(bed/sofa/kitchen/bath/desk/wardrobe),xCm,yCm,widthCm,lengthCm,rotationDeg}]}]}'
        user_prompt = f'Target Floor Level: {req.floor_level}.\nUser Request: {req.prompt}\nDesign EVERY requested room with accurate dimensions, flush shared walls, aligned interconnecting doors on shared walls, exterior-only windows, and furniture kept clear of all doorways.'
        g_key = os.environ.get('GEMINI_API_KEY', '').strip()
        if g_key:
            for gm, tl, tm in (('gemini-3.8-flash', 'low', 22), ('gemini-3.5-flash-lite', 'minimal', 18)):
                if plan_json_str: break
                try:
                    gurl = f'https://generativelanguage.googleapis.com/v1beta/models/{gm}:generateContent'
                    gpayload = {'systemInstruction': {'parts': [{'text': system_instruction + hint}]}, 'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}], 'generationConfig': {'responseMimeType': 'application/json', 'maxOutputTokens': 8192, 'thinkingConfig': {'thinkingLevel': tl}}}
                    gr = requests.post(gurl, headers={'x-goog-api-key': g_key, 'Content-Type': 'application/json'}, json=gpayload, timeout=tm)
                    if gr.status_code >= 400:
                        print(f'[AI_FALLBACK] Gemini:{gm} HTTP {gr.status_code}: {gr.text[:160]}', flush=True); continue
                    parts = gr.json()['candidates'][0]['content']['parts']
                    raw_g = ''.join(pt.get('text', '') for pt in parts if not pt.get('thought'))
                    plan_json_str = clean_plan(raw_g)
                    print(f'[AI_SUCCESS] Gemini:{gm} ({tl})', flush=True); break
                except HTTPException:
                    raise
                except Exception as e:
                    print(f'[AI_FALLBACK] Gemini:{gm} failed: {e}', flush=True)
        providers = []
        gk = os.environ.get('GROQ_API_KEY', '').strip()
        if gk:
            for gm in ('openai/gpt-oss-120b', 'openai/gpt-oss-20b'):
                providers.append(('Groq:' + gm, 'https://api.groq.com/openai/v1/chat/completions', gk, gm, False, 'low', 20))
        ork = os.environ.get('OPENROUTER_API_KEY', '').strip()
        if ork:
            providers.append(('OpenRouter:openrouter/free', 'https://openrouter.ai/api/v1/chat/completions', ork, 'openrouter/free', False, None, 20))
        for pname, purl, pkey, pmodel, use_rf, reff, pto in providers:
            if plan_json_str: break
            try:
                payload = {'model': pmodel, 'messages': [{'role': 'system', 'content': system_instruction + hint}, {'role': 'user', 'content': user_prompt}], 'max_tokens': 8192}
                if use_rf: payload['response_format'] = {'type': 'json_object'}
                if reff: payload['reasoning_effort'] = reff
                pr_resp = requests.post(purl, headers={'Authorization': f'Bearer {pkey}', 'Content-Type': 'application/json'}, json=payload, timeout=pto)
                if pr_resp.status_code >= 400:
                    print(f'[AI_FALLBACK] {pname} HTTP {pr_resp.status_code}: {pr_resp.text[:160]}', flush=True); continue
                msg = pr_resp.json()['choices'][0]['message']
                raw_out = msg.get('content') or msg.get('reasoning') or ''
                plan_json_str = clean_plan(raw_out)
                print(f'[AI_SUCCESS] {pname}', flush=True); break
            except HTTPException:
                raise
            except Exception as ex:
                print(f'[AI_FALLBACK] {pname} error: {ex}', flush=True); continue
        if not plan_json_str:
            raise HTTPException(status_code=400, detail='AI generation failed. Please try again.')
        daily, bonus = wallet['dailyCredits'], wallet['restoredBonusCredits']
        if daily >= 10: daily -= 10
        else: rem = 10 - daily; daily = 0; bonus -= rem
        with conn.cursor() as cur:
            cur.execute('UPDATE users SET daily_credits = %s, restored_bonus_credits = %s WHERE email = %s', (daily, bonus, email))
        updated_wallet = get_and_sync_wallet(conn, email)
        return {'wallet': updated_wallet, 'plan_json': plan_json_str}

