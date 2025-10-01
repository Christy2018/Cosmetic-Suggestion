# routers/products.py
from fastapi import APIRouter, Request, Depends, HTTPException, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import os
from uuid import uuid4

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Helper function to get the MongoDB database
async def get_db(request: Request):
    return request.app.mongodb

# Product Model
class Product(BaseModel):
    name: str
    brand: str
    price: float
    specs: str
    usage: str
    image_url: str = ""
    suitable_for: list = []

# Get All Products
@router.get("/api/products")
async def get_products(db = Depends(get_db)):
    try:
        products = await db.products.find().to_list(length=100)
        # Convert ObjectId to string for JSON serialization
        for product in products:
            product["_id"] = str(product["_id"])
        return {"products": products}
    except Exception as e:
        # Return mock data if database is not connected
        products = [
            {
                "id": "1",
                "name": "Hydrating Serum",
                "brand": "SkinCare",
                "price": 29.99,
                "specs": "Hyaluronic Acid Formula",
                "usage": "Apply morning and night to clean skin",
                "image_url": "/static/images/product1.jpg",
                "suitable_for": ["dry", "normal"]
            },
            {
                "id": "2",
                "name": "Oil Control Moisturizer",
                "brand": "ClearSkin",
                "price": 24.99,
                "specs": "Oil-free, non-comedogenic",
                "usage": "Apply twice daily to face and neck",
                "image_url": "/static/images/product2.jpg",
                "suitable_for": ["oily", "combination"]
            }
        ]
        return {"products": products}

# Get Single Product
@router.get("/api/products/{product_id}")
async def get_product(product_id: str, db = Depends(get_db)):
    try:
        product = await db.products.find_one({"_id": ObjectId(product_id)})
        if product:
            product["_id"] = str(product["_id"])
            return product
        raise HTTPException(status_code=404, detail="Product not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add New Product
@router.post("/api/products")
async def add_product(
    request: Request,
    name: str = Form(...),
    brand: str = Form(...),
    price: float = Form(...),
    specs: str = Form(...),
    usage: str = Form(...),
    suitable_for: str = Form(""),
    image: UploadFile = File(None),
    db = Depends(get_db)
):
    try:
        # Handle image upload if provided
        image_url = ""
        if image and image.filename:
            # Create uploads directory if it doesn't exist
            os.makedirs("static/uploads", exist_ok=True)
            
            # Generate unique filename to prevent overwrites
            file_extension = os.path.splitext(image.filename)[1]
            new_filename = f"{uuid4()}{file_extension}"
            file_path = f"static/uploads/{new_filename}"
            
            # Save the uploaded file
            with open(file_path, "wb") as f:
                content = await image.read()
                f.write(content)
            
            image_url = f"/static/uploads/{new_filename}"
        
        # Process suitable_for as a list
        suitable_for_list = []
        if suitable_for:
            suitable_for_list = [skin_type.strip() for skin_type in suitable_for.split(",")]
        
        # Create product object
        product = {
            "name": name,
            "brand": brand,
            "price": price,
            "specs": specs,
            "usage": usage,
            "image_url": image_url,
            "suitable_for": suitable_for_list
        }
        
        # Insert into database
        result = await db.products.insert_one(product)
        
        # Return created product with ID
        product["_id"] = str(result.inserted_id)
        return {"message": "Product added successfully", "product": product}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update Product
@router.put("/api/products/{product_id}")
async def update_product(
    product_id: str,
    request: Request,
    name: str = Form(...),
    brand: str = Form(...),
    price: float = Form(...),
    specs: str = Form(...),
    usage: str = Form(...),
    suitable_for: str = Form(""),
    image: UploadFile = File(None),
    db = Depends(get_db)
):
    try:
        # Check if product exists
        existing_product = await db.products.find_one({"_id": ObjectId(product_id)})
        if not existing_product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Handle image upload if provided
        image_url = existing_product.get("image_url", "")
        if image and image.filename:
            # Create uploads directory if it doesn't exist
            os.makedirs("static/uploads", exist_ok=True)
            
            # Generate unique filename to prevent overwrites
            file_extension = os.path.splitext(image.filename)[1]
            new_filename = f"{uuid4()}{file_extension}"
            file_path = f"static/uploads/{new_filename}"
            
            # Save the uploaded file
            with open(file_path, "wb") as f:
                content = await image.read()
                f.write(content)
            
            # Delete old image if exists and not a default image
            old_image_path = existing_product.get("image_url", "")
            if old_image_path and old_image_path.startswith("/static/uploads/"):
                try:
                    os.remove(old_image_path.lstrip("/"))
                except:
                    pass
            
            image_url = f"/static/uploads/{new_filename}"
        
        # Process suitable_for as a list
        suitable_for_list = []
        if suitable_for:
            suitable_for_list = [skin_type.strip() for skin_type in suitable_for.split(",")]
        
        # Update product object
        product = {
            "name": name,
            "brand": brand,
            "price": price,
            "specs": specs,
            "usage": usage,
            "image_url": image_url,
            "suitable_for": suitable_for_list
        }
        
        # Update in database
        await db.products.update_one(
            {"_id": ObjectId(product_id)},
            {"$set": product}
        )
        
        # Return updated product
        product["_id"] = product_id
        return {"message": "Product updated successfully", "product": product}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete Product
@router.delete("/api/products/{product_id}")
async def delete_product(product_id: str, db = Depends(get_db)):
    try:
        # Get product to check for image
        product = await db.products.find_one({"_id": ObjectId(product_id)})
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Delete associated image if it exists and is not a default image
        image_path = product.get("image_url", "")
        if image_path and image_path.startswith("/static/uploads/"):
            try:
                os.remove(image_path.lstrip("/"))
            except:
                pass
        
        # Delete from database
        result = await db.products.delete_one({"_id": ObjectId(product_id)})
        
        if result.deleted_count:
            return {"message": "Product deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Product not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Render Product Templates
@router.get("/products")
async def list_products_page(request: Request):
    return templates.TemplateResponse("products/list.html", {"request": request})

@router.get("/products/add")
async def add_product_page(request: Request):
    return templates.TemplateResponse("products/add.html", {"request": request})

@router.get("/products/{product_id}/edit")
async def edit_product_page(request: Request, product_id: str, db = Depends(get_db)):
    try:
        product = await db.products.find_one({"_id": ObjectId(product_id)})
        if product:
            product["_id"] = str(product["_id"])
            return templates.TemplateResponse(
                "products/edit.html", 
                {"request": request, "product": product}
            )
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": "Product not found"}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": str(e)}
        )