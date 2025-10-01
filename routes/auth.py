from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, RedirectResponse
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from pymongo import MongoClient
import uuid
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# User schema
class UserRegister(BaseModel):
    name: str  
    email: EmailStr
    password: str = Field(..., min_length=8)
    age: int
    gender: str
    skin_type: str
    skin_conditions: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    role: str  # Role is required in login

# Get MongoDB database
def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    db = client.cos
    print(db.cos)
    return db

# Hash password function
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

@router.get("/auth/login")
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/auth/register")
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.get("/auth/forgot_password")
async def forgot_password_page(request: Request):
    """Serve the Forgot Password page."""
    return templates.TemplateResponse("forgot_password.html", {"request": request})

@router.post("/auth/forgot_password")
async def forgot_password(email: str = Form(...)):
    """Process forgot password request."""
    db = get_db()
    user = db.users.find_one({"email": email.lower()})

    if not user:
        return JSONResponse(content={"detail": "Email not found"}, status_code=404)

    # Generate a unique reset token
    reset_token = str(uuid.uuid4())
    reset_link = f"http://127.0.0.1:8000/auth/reset_password?token={reset_token}"

    # Save the reset token in the database
    db.users.update_one({"email": email.lower()}, {"$set": {"reset_token": reset_token}})
    print(f"✅ Reset token generated for {email}: {reset_token}")

    # Email configuration
    conf = ConnectionConfig(
        MAIL_USERNAME="cosmeticsuggestion@gmail.com",
        MAIL_PASSWORD="pznldqzogesfkvrm",
        MAIL_FROM="cosmeticsuggestion@gmail.com",
        MAIL_PORT=587,
        MAIL_SERVER="smtp.gmail.com",
        MAIL_STARTTLS=True,
        MAIL_SSL_TLS=False,
        USE_CREDENTIALS=True
    )

    # Email message
    message = MessageSchema(
        subject="Password Reset Request",
        recipients=[email],
        body=f"Click the link below to reset your password:\n\n{reset_link}",
        subtype="plain"
    )

    # Send the email
    try:
        fm = FastMail(conf)
        await fm.send_message(message)
        print(f"✅ Reset link sent to {email}")
        return JSONResponse(content={"message": "Reset link sent to your email", "reset_link": reset_link}, status_code=200)
    except Exception as e:
        print("❌ Error sending email:", str(e))
        return JSONResponse(content={"error": "Failed to send reset link", "details": str(e)}, status_code=500)

from starlette.responses import RedirectResponse

@router.post("/auth/login")
async def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    
):
    
    email = email.lower()
    db=get_db()
    users = db.users
    user = users.find_one({"email": email})
    print(email)

    if not user:
        return JSONResponse(content={"detail": "Invalid email or password"}, status_code=401)

    # Check if the user is blocked
    if user.get("status") == "blocked":
        return JSONResponse(content={"detail": "Your account is blocked. Please contact support."}, status_code=403)

    # Verify password
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        return JSONResponse(content={"detail": "Invalid email or password"}, status_code=401)

    # Check role
    stored_role = user.get("role", "user")
    if stored_role != role:
        return JSONResponse(content={"detail": "Incorrect role selected"}, status_code=403)

    # Store user email and role in the session
    request.session["user_email"] = email
    request.session["user_role"] = stored_role
    print("Session set: user_email =", request.session["user_email"], "user_role =", request.session["user_role"])

    # Determine redirection URL
    redirect_url = "/admin-dashboard" if stored_role == "admin" else "/user-dashboard"

    # Perform actual redirection
    return RedirectResponse(url=redirect_url, status_code=302)


@router.get("/auth/logout")
async def logout():
    return JSONResponse(content={"message": "Logged out successfully"}, status_code=200)
