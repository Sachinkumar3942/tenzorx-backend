from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

# Import our refactored model service
import model_service

app = FastAPI(title="TenZorX Underwriting API", version="1.0")

# Configure CORS so our Next.js frontend can communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For hackathon, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_loan(
    images: List[UploadFile] = File(...),
    video: Optional[UploadFile] = File(None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    shop_size_sqft: Optional[int] = Form(150),
    answers: Optional[str] = Form(None) # JSON string if needed
):
    try:
        # Validate number of images
        if len(images) < 3 or len(images) > 5:
            raise HTTPException(status_code=400, detail="Please upload between 3 and 5 images.")

        # Read images into memory
        image_bytes_list = []
        for img in images:
            image_bytes_list.append(await img.read())

        # Evaluate the loan using our model
        result = model_service.evaluate_loan(
            images_bytes=image_bytes_list, 
            lat=latitude, 
            lon=longitude,
            shop_size_sqft=shop_size_sqft
        )
        
        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
