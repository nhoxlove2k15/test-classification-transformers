from typing import List
from pydantic import BaseModel

class OneThesisPredictionRequest(BaseModel):
        test_ids: list
	

class ThesesPredictionRequest(BaseModel):
	books: List[OneThesisPredictionRequest] = []