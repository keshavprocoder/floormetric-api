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
    subject = f"{otp} is your Floor Metric verification code"
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:460px;margin:auto;padding:24px;border:1px solid #e2e8f0;border-radius:12px;">
      <h2 style="margin:0 0 12px;color:#0f172a;">Floor Metric Verification</h2>
      <p style="color:#334155;font-size:14px;">Use the 6-digit code below to sign in to Floor Metric. This code expires in {OTP_EXPIRY_MINUTES} minutes.</p>
      <div style="font-size:32px;font-weight:bold;letter-spacing:8px;color:#0284c7;background:#f0f9ff;padding:16px;text-align:center;border-radius:8px;margin:20px 0;">
        {otp}
      </div>
      <p style="color:#64748b;font-size:12px;">If you did not request this code, you can safely ignore this email.</p>
    </div>
    """

    gas_url = os.environ.get("GAS_EMAIL_URL", "").strip()
    if gas_url:
        resp = requests.post(
            gas_url,
            json={
                "secret": os.environ.get("GAS_EMAIL_SECRET", "").strip(),
                "to": to_email,
                "subject": subject,
                "html": html_body,
            },
            timeout=15,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail="Failed to send verification email.")
        return

    brevo_key = os.environ.get("BREVO_API_KEY", "").strip()
    sender_email = os.environ.get("SENDER_EMAIL", "").strip()
    if brevo_key and sender_email:
        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "api-key": brevo_key,
                "Content-Type": "application/json",
                "accept": "application/json",
            },
            json={
                "sender": {"name": "Floor Metric", "email": sender_email},
                "to": [{"email": to_email}],
                "subject": subject,
                "htmlContent": html_body,
            },
            timeout=15,
        )
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail="Failed to send verification email via Brevo.")
        return

    raise HTTPException(
        status_code=500,
        detail="Email service not configured. Add GAS_EMAIL_URL or BREVO_API_KEY in Render.",
    )


# ── Wallet Helpers (Supabase PostgreSQL) ─────────────────────────
def get_and_sync_wallet(conn, email: str) -> dict:
    now_utc = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT has_lifetime_pass, daily_credits, yesterday_unused,
                   restored_bonus_credits, last_reset_utc
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
            }

        has_pro, daily, yest, bonus, last_reset = row
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

    _send_otp_email(email, otp)
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
                SET has_lifetime_pass = TRUE, daily_credits = 100, last_reset_utc = %s
                WHERE email = %s
                """,
                (now_utc, email),
            )
        return get_and_sync_wallet(conn, email)


@app.post("/restore-credits")
def restore_credits(
    req: VerifyPurchaseRequest,
    email: str = Depends(get_current_email),
):
    if not req.purchase_token or len(req.purchase_token) < 10:
        raise HTTPException(status_code=400, detail="Invalid Google Play purchase token")
    with get_db() as conn:
        wallet = get_and_sync_wallet(conn, email)
        restore_amt = wallet["yesterdayUnused"] if wallet["yesterdayUnused"] > 0 else 100
        new_bonus = wallet["restoredBonusCredits"] + restore_amt
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET yesterday_unused = 0, restored_bonus_credits = %s
                WHERE email = %s
                """,
                (new_bonus, email),
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
            raise HTTPException(
                status_code=402,
                detail="Insufficient credits. Restore credits (₹499).",
            )

        system_instruction = (
            "You are a senior residential architect. Convert the prompt into a realistic, "
            "non-overlapping 2D floor plan in centimeters. Stack rooms flush edge-to-edge so adjacent "
            "rooms share walls. Place doors on connecting walls and windows on exterior walls. "
            "Ensure all fixtures fit comfortably inside room boundaries."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Floor Level: {req.floor_level}. Design request: {req.prompt}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=HousePlanSchema,
                temperature=0.3,
            ),
        )

        daily = wallet["dailyCredits"]
        bonus = wallet["restoredBonusCredits"]
        if daily >= 10:
            daily -= 10
        else:
            rem = 10 - daily
            daily = 0
            bonus -= rem

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET daily_credits = %s, restored_bonus_credits = %s
                WHERE email = %s
                """,
                (daily, bonus, email),
            )
        updated_wallet = get_and_sync_wallet(conn, email)

        return {
            "wallet": updated_wallet,
            "plan_json": response.text,
        }
