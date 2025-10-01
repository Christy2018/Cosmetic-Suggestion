from fastapi import FastAPI, Request, Depends, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr
import bcrypt
import smtplib
import secrets
import shutil
import os
import imghdr  # ✅ To check if the file is an image
from datetime import datetime, timedelta

# Import routers
from routes import auth, blogs, users, products

# MongoDB connection setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.mongodb_client = AsyncIOMotorClient("mongodb://localhost:27017")
    app.mongodb = app.mongodb_client.cos
    print("✅ Connected to MongoDB")
    yield
    app.mongodb_client.close()
    print("❌ Closed MongoDB connection")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(auth.router)
app.include_router(blogs.router)
app.include_router(users.router)
app.include_router(products.router)

# Directory for uploaded images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin-dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/user-dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    return templates.TemplateResponse("user_dashboard.html", {"request": request})

@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse(url="/auth/login", status_code=303)
    return response

@app.get("/auth/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})


# **Login Request Model**
class LoginRequest(BaseModel):
    email: str
    password: str
    role: str  # Include role selection

# **Login Route**
@app.post("/auth/login")
async def login(
    request: Request, 
    email: str = Form(...), 
    password: str = Form(...), 
    role: str = Form(...)  
):
    db = request.app.mongodb
    if not db:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    email = email.lower()
    user = await db.users.find_one({"email": email})

    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
        stored_role = user.get("role", "user")

        if stored_role != role:
            return JSONResponse(content={"detail": "Incorrect role selected"}, status_code=403)

        redirect_url = "/admin-dashboard" if stored_role == "admin" else "/user-dashboard"

        return JSONResponse(content={"message": "Login successful", "role": stored_role, "redirect": redirect_url}, status_code=200)

    return JSONResponse(content={"detail": "Invalid email or password"}, status_code=401)

# **Register Request Model**
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    age: int
    gender: str
    skin_type: str
    skin_condition: str = None

# **Register Route**
@app.post("/auth/register")
async def register(user: RegisterRequest, request: Request):
    try:
        db = request.app.mongodb
        if not db:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        email = user.email.lower()
        existing_user = await db.users.find_one({"email": email})
        if existing_user:
            return JSONResponse(content={"error": "Email already exists"}, status_code=400)
        
        hashed_password = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        
        new_user = user.dict()
        new_user["password"] = hashed_password
        new_user["role"] = "user"
        new_user["status"] = "active"
        
        await db.users.insert_one(new_user)

        # 📧 Send Welcome Email
        send_email(email, "Welcome to Our Platform", "Thank you for registering!")

        return JSONResponse(content={"message": "Registration successful!", "redirect": "/auth/login"}, status_code=201)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=422)

# 📧 **Function to Send Emails**
def send_email(receiver_email: str, subject: str, body: str):
    SENDER_EMAIL = os.getenv("EMAIL_USER")  # ✅ Use environment variables
    APP_PASSWORD = os.getenv("EMAIL_PASS")

    message = f"Subject: {subject}\n\n{body}"
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, message)
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", str(e))

# **Generate Password Reset Token**
async def generate_reset_token(email: str, db):
    token = secrets.token_urlsafe(32)  
    expiry = datetime.utcnow() + timedelta(hours=1)  

    await db.password_resets.insert_one({
        "email": email,
        "token": token,
        "expires_at": expiry
    })

    return token

# **Forgot Password Request**
@app.post("/auth/forgot-password")
async def forgot_password(request: Request, email: str = Form(...)):
    db = request.app.mongodb
    user = await db.users.find_one({"email": email.lower()})

    if not user:
        return JSONResponse(content={"error": "Email not registered"}, status_code=400)

    reset_token = await generate_reset_token(email.lower(), db)
    reset_link = f"http://localhost:8000/auth/reset-password?token={reset_token}"
    send_email(email, "Password Reset Request", f"Click the link to reset your password: {reset_link}")

    return JSONResponse(content={"message": "Password reset link sent! Check your email."}, status_code=200)

# **Reset Password**
@app.post("/auth/reset-password")
async def reset_password(
    request: Request, 
    token: str = Form(...), 
    new_password: str = Form(...)
):
    db = request.app.mongodb
    reset_entry = await db.password_resets.find_one({"token": token})

    if not reset_entry or reset_entry["expires_at"] < datetime.utcnow():
        return JSONResponse(content={"error": "Invalid or expired token"}, status_code=400)

    hashed_password = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    await db.users.update_one({"email": reset_entry["email"]}, {"$set": {"password": hashed_password}})
    await db.password_resets.delete_one({"token": token})

    return JSONResponse(content={"message": "Password reset successful! You can now log in."}, status_code=200)

# ✅ **Image Upload Route**
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"

    # ✅ Check if the file is a valid image
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if imghdr.what(file_location) is None:
        os.remove(file_location)
        return JSONResponse(content={"error": "Invalid image file"}, status_code=400)

    return JSONResponse(content={"message": "Image uploaded successfully", "file_path": file_location}, status_code=200)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)