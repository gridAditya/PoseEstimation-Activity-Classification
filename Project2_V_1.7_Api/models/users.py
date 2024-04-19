from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List

class User(BaseModel):
    username: str
    email: EmailStr
    password: str
    salt: str
    tokens: List[str] = []
    request_count: int = 0
    last_request_timestamp: datetime = None
    rate_limit_threshold: int = 7
    rate_limit_window: int = 60