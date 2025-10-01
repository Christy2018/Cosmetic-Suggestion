from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str = Field(..., min_length=8)
    age: int
    gender: str
    skin_type: str
    skin_conditions: Optional[str] = None

class ReviewCreate(BaseModel):
    predicted_condition: str
    recommendation_link: str
    review_text: str
    rating: int = Field(..., ge=1, le=5)  # Rating must be between 1 and 5

class ReviewResponse(BaseModel):
    user_email: EmailStr
    predicted_condition: str
    recommendation_link: str
    review_text: str
    rating: int
    timestamp: datetime