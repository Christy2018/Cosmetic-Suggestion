# routers/users.py
from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from db import get_database


router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Helper function to get the MongoDB database
from main import get_database

def get_db():
    return get_database()


# User Model for API
class UserUpdate(BaseModel):
    name: str = None
    email: str = None
    role: str = None
    status: str = None

# User Management Page
@router.get("/admin/user-management")
async def admin_user_management(request: Request):
    return templates.TemplateResponse("user_management.html", {"request": request})

# Get All Users
@router.get("/api/users") 
def get_users(db = Depends(get_db)):
    try:
        users = list(db.users.find())
        for user in users:
            user["_id"] = str(user["_id"])
        return {"users": users}
    except Exception as e:
        print("❌ Error fetching users:", str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch users from DB")

# Get User by ID
@router.get("/api/users/{user_id}")
async def get_user(user_id: str, db = Depends(get_db)):
    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if user:
            user["_id"] = str(user["_id"])
            return {"user": user}
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update User
@router.put("/api/users/{user_id}")
async def update_user(user_id: str, user: UserUpdate, db = Depends(get_db)):
    try:
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)}, 
            {"$set": {k: v for k, v in user.dict().items() if v is not None}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete User
@router.delete("/api/users/{user_id}")
async def delete_user(user_id: str, db = Depends(get_db)):
    try:
        result = await db.users.delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))