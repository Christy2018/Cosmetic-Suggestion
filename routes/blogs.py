# routers/blogs.py
from fastapi import APIRouter, HTTPException, Form, UploadFile, File, Depends, Request
from pydantic import BaseModel
from uuid import uuid4
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from db import get_database as get_db
from datetime import datetime


router = APIRouter()

# Helper function to get the MongoDB database

# Blog Model
class Blog(BaseModel):
    title: str
    content: str
    image_url: str = ""

# Get All Blogs
@router.get("/api/blogs")
def get_blogs(db=Depends(get_db)):
    blogs = list(db.blogs.find().sort("created_at", -1))
    return [
        {
            "id": str(blog["_id"]),  # Convert ObjectId to string
            "title": blog["title"],
            "content": blog["content"],
            "image_url": blog.get("image_url")
        }
        for blog in blogs
    ]


# Add New Blog
@router.post("/api/blogs/add")
async def add_blog(
    request: Request,
    title: str = Form(...), 
    content: str = Form(...), 
    image: UploadFile = File(None),
    db = Depends(get_db)
):
    try:
        blog_id = str(uuid4())
        image_url = ""

        # Create static/images directory if it doesn't exist
        if not os.path.exists("static/images"):
            os.makedirs("static/images")

        if image:
            image_path = f"static/images/{blog_id}_{image.filename}"
            with open(image_path, "wb") as f:
                f.write(await image.read())
            image_url = "/" + image_path

        blog = {
            "id": blog_id,
            "title": title,
            "content": content,
            "image_url": image_url,
            "created_at": datetime.utcnow()  # Add a timestamp
        }

        # Insert the blog into the database
        result = db.blogs.insert_one(blog)
        blog["_id"] = str(result.inserted_id)  # Convert ObjectId to string
        print("Blog inserted successfully:", blog)

        return {"message": "Blog added successfully!", "blog": blog}
    except Exception as e:
        print("❌ Error adding blog:", str(e))
        raise HTTPException(status_code=500, detail="Failed to add blog")

# Edit Blog
@router.put("/api/blogs/edit/{blog_id}")
async def edit_blog(blog_id: str, blog: Blog, db=Depends(get_db)):
    result = db.blogs.update_one({"id": blog_id}, {"$set": blog.dict()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Blog not found")
    return {"message": "Blog updated successfully!", "blog": blog.dict()}

# Delete Blog
@router.delete("/api/blogs/delete/{blog_id}")
async def delete_blog(blog_id: str, db=Depends(get_db)):
    try:
        # Convert blog_id to ObjectId if using MongoDB's default _id field
        result = db.blogs.delete_one({"_id": ObjectId(blog_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Blog not found")
        return {"message": "Blog deleted successfully!"}
    except Exception as e:
        print("❌ Error deleting blog:", str(e))
        raise HTTPException(status_code=500, detail="Failed to delete blog")