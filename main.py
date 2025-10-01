from fastapi import FastAPI, Request, Depends, HTTPException, Form, File, UploadFile, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from pydantic import BaseModel, EmailStr
import bcrypt
import shutil
import os
import imghdr
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as tfk
from starlette.middleware.sessions import SessionMiddleware
import pandas as pd
import asyncio
from db import get_database
from fastapi import Body
import uuid
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

# **Register Request Model**
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    age: int
    gender: str
    skin_type: str
    skin_condition: str = None

# **Login Request Model**
class LoginRequest(BaseModel):
    email: str
    password: str
    role: str  

# Import routers
from routes import auth, blogs, users, products

# **Model Initialization**
model = None

def initialize_model():
    global model
    img_size = (224, 224)
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    mobileNet = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/tf2-preview-feature-vector/4"
    mNet = hub.KerasLayer(mobileNet, input_shape=img_size + (3,), trainable=False)

    model = tfk.Sequential([
        mNet,
        tfk.layers.Dense(170, activation='relu'),
        tfk.layers.Dense(8, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    print("Model training and evaluation completed.")

def cos_model(img_path):
    labels = {
        "BA-cellulitis": 0,
        "BA-impetigo": 1,
        "FU-athlete-foot": 2,
        "FU-nail-fungus": 3,
        "FU-ringworm": 4,
        "PA-cutaneous-larva-migrans": 5,
        "VI-chickenpox": 6,
        "VI-shingles": 7
    }  # Replace with actual labels
    inv_labels = {v: k for k, v in labels.items()}
    img_size = (224, 224)

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found or unable to load. Check the file path:", img_path)
    print("loaded-1")
    img_resized = cv2.resize(img, img_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    print("loaded-2")
    predictions = model.predict(img_input)
    print(predictions)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = inv_labels[predicted_index]
    print("Predicted label:", predicted_label)
    return predicted_label

@asynccontextmanager
async def model_lifespan(app: FastAPI):
    global model
    try:
        print("Initializing model during startup...")
        await asyncio.to_thread(initialize_model)
        print("Model initialized successfully.")
        yield
    except Exception as e:
        print("❌ Error in model lifespan:", str(e))
    finally:
        print("Shutting down model...")
        model = None
        print("Model resources cleaned up.")

# **FastAPI Initialization**
app = FastAPI(lifespan=model_lifespan)
app.add_middleware(SessionMiddleware, secret_key="0ac65b9f346669416fc60bbfee03624ac71b6fe85474d8f08c89667337d0ef5b")

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
    print("Rendering admin_dashboard.html")  # Debug log
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

@app.get("/auth/reset_password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str):
    return templates.TemplateResponse("reset_password.html", {"request": request, "token": token})

@app.get("/admin/user-reports/")
async def user_reports_page(request: Request):
    reports = ["Report 1", "Report 2", "Report 3"]  # Replace with actual data
    return templates.TemplateResponse("user_reports.html", {"request": request, "reports": reports})

@app.get("/admin/blog-management", response_class=HTMLResponse)
async def admin_blog_management(request: Request):
    return templates.TemplateResponse("manage_blogs.html", {"request": request})

@app.get("/user/blogs", response_class=HTMLResponse)
async def user_blogs(request: Request):
    return templates.TemplateResponse("view_blog.html", {"request": request})


# Initialize the router
router = APIRouter()

# Include the router
app.include_router(router)

# **Endpoints**
@app.post("/api/analyse-image")
async def analyse_image(request: Request, file: UploadFile = File(...)):
    try:
        print("✅ Received file:", file.filename)

        user_email = request.session.get("user_email")
        if not user_email:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Generate a unique name for the uploaded image
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join("uploaded_images", unique_filename)

        # Save the uploaded image
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("✅ File saved at:", file_path)

        # Validate the uploaded file is an image
        if imghdr.what(file_path) is None:
            os.remove(file_path)  # Remove invalid file
            print("❌ Invalid image file")
            return JSONResponse(content={"error": "Invalid image file"}, status_code=400)

        # Predict the label using the model
        predicted_label = cos_model(file_path)  # Pass the dynamically generated file path
        print("✅ Predicted label:", predicted_label)

        # Normalization function
        def normalize(label):
            return label.strip().lower().replace("_", "-").replace(" ", "-")

        # Manual mapping for mismatches between prediction and Excel
        manual_label_mapping = {
            "vi-chickenpox": "vi-chicken-pox",
            "fu-ringworm": "fu-ring-worm"
        }

        # Load the Excel file
        excel_file = "disease_cosmetics_link.xlsx"
        disease_links_df = pd.read_excel(excel_file)

        # Normalize Excel data
        disease_links = {
            normalize(row["Disease"]): row["Product_link"]
            for _, row in disease_links_df.iterrows()
        }

        # Apply normalization and mapping
        normalized_label = normalize(predicted_label)
        normalized_label = manual_label_mapping.get(normalized_label, normalized_label)
        link = disease_links.get(normalized_label, None)

        print("✅ Recommendation link:", link if link else "None found")

        if not link:
            print("❌ No match found for:", normalized_label)
            print("🗂️ Excel keys:", list(disease_links.keys()))

        # Insert into history
        db = get_database()
        history_entry = {
            "user_email": user_email,
            "prediction": predicted_label,
            "recommendation": link,
            "date": datetime.now()
        }
        db.history.insert_one(history_entry)
        print("📦 History entry added:", history_entry)

        # Remove the uploaded image after analysis
        try:
            os.remove(file_path)
            print("✅ Uploaded image removed:", file_path)
        except Exception as e:
            print("❌ Error removing uploaded image:", str(e))

        # Return the response with the disease name in bold
        if link:
            return JSONResponse(content={
                "success": True,
                "disease": f"<b>{predicted_label}</b>",  # Make the disease name bold
                "link": link
            })
        else:
            return JSONResponse(content={
                "success": True,
                "disease": f"<b>{predicted_label}</b>",  # Make the disease name bold
                "link": None,
                "message": "No recommendation available for this condition yet."
            })
    except Exception as e:
        print("❌ Error:", str(e))
        return JSONResponse(content={"message": "Error processing image", "error": str(e)}, status_code=500)

@app.get("/recommendation/{disease_name}")
async def get_recommendation(disease_name: str):
    excel_file = "disease_cosmetics_link.xlsx"
    disease_links_df = pd.read_excel(excel_file)
    disease_links = {row["Disease"].lower(): row["Product_link"] for _, row in disease_links_df.iterrows()}
    disease_name = disease_name.lower()
    if disease_name in disease_links:
        link = disease_links[disease_name]
        return {"disease": disease_name, "link": link}
    else:
        raise HTTPException(status_code=404, detail=f"Disease '{disease_name}' not found")

@app.get("/products/{product_name}", response_class=HTMLResponse)
async def get_product_page(request: Request, product_name: str):
    try:
        # Ensure the product_name does not already include ".html"
        if not product_name.endswith(".html"):
            product_name += ".html"
        # Render the product page dynamically
        return templates.TemplateResponse(f"products/{product_name}", {"request": request})
    except Exception as e:
        print(f"❌ Error loading product page: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.get("/api/user_history")
async def get_user_history(request: Request, db=Depends(get_database)):
    user_email = request.session.get("user_email")  # Ensure session stores email
    
    if not user_email:
        raise HTTPException(status_code=401, detail="User not authenticated")

    print("Fetching history for user:", user_email)  # Debug log

    # Fetch user history from the database
    history = list(db.history.find({"user_email": user_email}).sort("date", -1))
    print("Fetched history:", history)  # Debug log

    # Format the response
    formatted_history = [
        {
            "date": entry["date"].strftime("%Y-%m-%d %H:%M:%S"),  # Format the date
            "prediction": entry["prediction"],
            "recommendation": entry["recommendation"]
        }
        for entry in history
    ]
    
    return JSONResponse(content=formatted_history, status_code=200)

@app.get("/api/reports")
async def get_reports(db=Depends(get_database)):
    try:
        history_records = list(db.history.find({}).sort("date", -1))

        reports = []
        for record in history_records:
            email = record.get("user_email", "N/A")
            disease = record.get("prediction", "N/A")
            timestamp = record.get("date")

            # Fetch user name (optional, based on email)
            user = db.users.find_one({"email": email})
            name = user.get("name", "Unknown") if user else "Unknown"

            reports.append({
                "email": email,
                "name": name,
                "disease_reported": disease,
                "timestamp": timestamp
            })

        return {"reports": reports}
    except Exception as e:
        print("Error in /api/reports:", str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch reports")

@app.get("/api/reviews/all")
async def get_all_reviews(db=Depends(get_database)):
    try:
        reviews = list(db.reviews.find().sort("timestamp", -1))

        formatted_reviews = []
        for r in reviews:
            formatted_reviews.append({
                "user_email": r.get("user_email", "N/A"),
                "comment": r.get("review_text", "No comment provided."),
                "rating": r.get("rating", "N/A"),
                "timestamp": r.get("timestamp")
            })

        return formatted_reviews
    except Exception as e:
        print("❌ Error fetching all reviews:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
from datetime import datetime
import pytz
from schemas import ReviewCreate
from models import Review
from schemas import ReviewResponse

@app.post("/api/review")
async def submit_review(
    review: ReviewCreate,
    request: Request,
    db=Depends(get_database)
):
    try:
        print("Incoming review data:", review.dict())

        user_email = request.session.get("user_email")
        print("Session user_email:", user_email)
        if not user_email:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Set the desired timezone (e.g., "Asia/Kolkata" for Indian Standard Time)
        local_timezone = pytz.timezone("Asia/Kolkata")
        current_time = datetime.now(local_timezone)

        review_data = Review(
            user_email=user_email,
            predicted_condition=review.predicted_condition,
            recommendation_link=review.recommendation_link,
            review_text=review.review_text,
            rating=review.rating,
            timestamp=current_time  # Use the localized timestamp
        ).dict()
        print("Review data to insert:", review_data)

        # Corrected: Remove `await` from insert_one
        result = db.reviews.insert_one(review_data)
        print("Insert result:", result.inserted_id)

        return {"message": "Review submitted successfully"}
    except Exception as e:
        print("Error in /api/review:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/api/reviews/{predicted_condition}", response_model=list[ReviewResponse])
async def get_reviews(predicted_condition: str, request: Request, db=Depends(get_database)):
    # Get the logged-in user's email from the session
    user_email = request.session.get("user_email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User not authenticated")

    print("Predicted condition:", predicted_condition)
    print("User email from session:", user_email)

    # Build the query to filter reviews
    query = {"user_email": user_email}
    if predicted_condition != "all":
        query["predicted_condition"] = predicted_condition

    print("Query:", query)  # Debug log

    # Fetch reviews from the database
    reviews = db.reviews.find(query).to_list(None)
    print("Fetched reviews:", reviews)  # Debug log

    return reviews

@app.get("/api/reviews", response_model=list[dict])
async def get_all_reviews(db=Depends(get_database)):
    """
    Fetch all user reviews for the admin dashboard.
    """
    reviews = db.reviews.find().to_list(None)
    return [
        {
            "user_name": review.get("user_name", "Anonymous"),
            "comment": review.get("comment", "No comment provided."),
            "rating": review.get("rating", 0)
        }
        for review in reviews
    ]

@router.get("/api/blogs")
def get_blogs(db=Depends(get_database)):  # ✅ remove async
    blogs = list(db.blogs.find().sort("created_at", -1))  # ✅ remove await
    return [
        {
            "id": str(blog["_id"]),
            "title": blog["title"],
            "content": blog["content"],
            "image_url": blog.get("image_url")
        }
        for blog in blogs
    ]


@app.delete("/api/blogs/delete/{id}")
async def delete_blog(id: str, db=Depends(get_database)):
    try:
        result = db.blogs.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Blog not found")
        return {"message": "Blog deleted successfully"}
    except Exception as e:
        print("❌ Error deleting blog:", str(e))
        raise HTTPException(status_code=500, detail="Failed to delete blog")


@app.post("/logout")
async def logout():
    return JSONResponse(content={"message": "Logged out successfully"})

@app.post("/auth/register")
async def register(user: RegisterRequest, ):
    try:
        db=get_database()
        print("Registering user:", user.email)
        email = user.email.lower()
        print("Recipient email:", email)
        existing_user = db.users.find_one({"email": email})
        if existing_user:
            print("Email already exists:", email)
            return JSONResponse(content={"error": "Email already exists"}, status_code=400)

        hashed_password = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        new_user = user.dict()
        new_user["password"] = hashed_password
        new_user["role"] = "user"
        new_user["status"] = "active"

        db.users.insert_one(new_user)
        print("User registered successfully:", email)
        return JSONResponse(content={"message": "Registration successful!", "redirect": "/auth/login"}, status_code=201)
    except Exception as e:
        print("Error during registration:", str(e))
        return JSONResponse(content={"error": "Registration failed", "details": str(e)}, status_code=500)

@app.post("/auth/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
):
    try:
        db = get_database()
        print("Login attempt for email:", email)
        user = db.users.find_one({"email": email.lower()})
        if user:
            print("User found:", user["email"])
            if bcrypt.checkpw(password.encode("utf-8"), user["password"].encode("utf-8")):
                # Check if the user is blocked
                if user.get("status") == "blocked":
                    print("Blocked user attempted login:", email)
                    return JSONResponse(content={"detail": "Your account has been blocked. Please contact admin."}, status_code=403)

                # Check if the role matches
                if user.get("role") != role:
                    print("Invalid role selected for user:", email)
                    return JSONResponse(content={"detail": "Invalid role selected"}, status_code=400)

                # Set session data
                request.session["user_email"] = email.lower()
                request.session["user_role"] = role
                print("Login successful for user:", email)
                return RedirectResponse(url="/user-dashboard", status_code=302)
            else:
                print("Invalid password for user:", email)
        else:
            print("User not found:", email)

        return JSONResponse(content={"detail": "Invalid email or password"}, status_code=401)
    except Exception as e:
        print("Error during login:", str(e))
        return JSONResponse(content={"error": "Login failed", "details": str(e)}, status_code=500)
    
    from fastapi import UploadFile, File
import uuid

@app.post("/api/blogs/add")
async def add_blog(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    image: UploadFile = File(None),
    db=Depends(get_database)
):
    try:
        image_url = None

        # Save image if uploaded
        if image:
            upload_dir = "static/uploads/blog_images"
            os.makedirs(upload_dir, exist_ok=True)

            filename = f"{uuid.uuid4().hex}_{image.filename}"
            file_path = os.path.join(upload_dir, filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            image_url = f"/static/uploads/blog_images/{filename}"

        blog = {
            "title": title,
            "content": content,
            "image_url": image_url,
            "created_at": datetime.utcnow()
        }

        db.blogs.insert_one(blog)
        return {"message": "Blog added successfully"}

    except Exception as e:
        print("❌ Error adding blog:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-db")
async def test_db():
    try:
        db=get_database()
        collections = db.list_collection_names()
        return {"collections": collections}
    except Exception as e:
        print("❌ Database test failed:", str(e))
        return {"error": "Database connection failed", "details": str(e)}

@app.get("/api/users")
def get_users(
    page: int = 1,
    limit: int = 10,
    search: str = None,
    status: str = None,
    db=Depends(get_database)
):
    try:
        query = {"role": "user"}  # Only fetch users, exclude admins

        # Add search filter
        if search:
            query["$or"] = [
                {"name": {"$regex": search, "$options": "i"}},  # Case-insensitive search for name
                {"email": {"$regex": search, "$options": "i"}}  # Case-insensitive search for email
            ]

        # Add status filter
        if status:
            query["status"] = status

        print("Query being used:", query)  # Debug log

        # Fetch filtered users
        users = list(db.users.find(query).skip((page - 1) * limit).limit(limit))
        total_users = db.users.count_documents(query)

        # Format the response
        formatted_users = [
            {
                "name": user.get("name", "N/A"),
                "email": user.get("email", "N/A"),
                "role": user.get("role", "N/A"),
                "status": user.get("status", "N/A")
            }
            for user in users
        ]
        print("📦 Returning users:", formatted_users)

        return {"users": formatted_users, "total": total_users}
    except Exception as e:
        print("Error fetching users:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.put("/api/users/{email}/status")
def update_user_status(email: str, data: dict = Body(...), db=Depends(get_database)):
    try:
        status = data.get("status")
        if status not in ["active", "blocked"]:
            raise HTTPException(status_code=400, detail="Invalid status")

        user = db.users.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.get("role") == "admin":
            raise HTTPException(status_code=403, detail="Cannot update admin accounts")

        db.users.update_one({"email": email}, {"$set": {"status": status}})
        return {"message": f"User status updated to {status}"}
    except Exception as e:
        print("Error updating user status:", str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/auth/forgot_password")
async def forgot_password(email: str = Form(...), db=Depends(get_database)):
    try:
        # Check if the email exists in the database
        user = db.users.find_one({"email": email.lower()})
        if not user:
            raise HTTPException(status_code=404, detail="Email not found")

        # Generate a unique reset token
        reset_token = str(uuid.uuid4())
        reset_link = f"http://127.0.0.1:8000/auth/reset_password?token={reset_token}"

        # Save the reset token in the database
        db.users.update_one({"email": email.lower()}, {"$set": {"reset_token": reset_token}})
        print(f"✅ Reset token generated for {email}: {reset_token}")

        # Send the reset link via email
        message = MessageSchema(
            subject="Password Reset Request",
            recipients=[email],
            body=f"Click the link below to reset your password:\n\n{reset_link}",
            subtype="plain"
        )
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
        fm = FastMail(conf)
        print("✅ Sending email...")
        await fm.send_message(message)
        print(f"✅ Reset link sent to {email}")

        return {"message": "Password reset link sent to your email.", "reset_link": reset_link}
    except Exception as e:
        print("❌ Error in forgot password:", str(e))
        raise HTTPException(status_code=500, detail="Failed to process forgot password request")

@app.post("/auth/reset_password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db=Depends(get_database)
):
    try:
        if new_password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")

        # Find the user with the reset token
        user = db.users.find_one({"reset_token": token})
        if not user:
            raise HTTPException(status_code=404, detail="Invalid or expired reset token")

        # Hash the new password
        hashed_password = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Update the user's password and remove the reset token
        db.users.update_one(
            {"reset_token": token},
            {"$set": {"password": hashed_password}, "$unset": {"reset_token": ""}}
        )
        print(f"✅ Password reset successfully for {user['email']}")

        return {"message": "Password reset successfully. You can now log in with your new password."}
    except Exception as e:
        print("❌ Error in reset password:", str(e))
        raise HTTPException(status_code=500, detail="Failed to reset password")

@app.get("/test-email")
async def test_email():
    try:
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

        message = MessageSchema(
            subject="Test Email from FastAPI",
            recipients=["christychacko2018@gmail.com"],  # Replace with your email
            body="This is a test email sent using FastAPI-Mail 🎉",
            subtype="plain"
        )

        fm = FastMail(conf)
        await fm.send_message(message)
        return {"message": "Email sent successfully"}
    except Exception as e:
        print("❌ Error sending test email:", str(e))
        return {"error": str(e)}

@app.get("/test-api/")
async def test_api():
    return {"message": "Here is the latest cosmetic tip!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)