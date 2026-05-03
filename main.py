from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
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
async def predict_loan(request: Request):
    try:
        # Parse multipart form data
        form = await request.form()
        
        # Extract images
        images = form.getlist("images")
        if len(images) < 3 or len(images) > 5:
            raise HTTPException(status_code=400, detail="Please upload between 3 and 5 images.")

        # Read images into memory and extract metadata
        image_bytes_list = []
        image_lats = []
        image_lons = []
        
        for idx, img_file in enumerate(images):
            image_bytes_list.append(await img_file.read())
            
            # Extract capture location metadata from form
            lat_key = f"image_{idx}_lat"
            lon_key = f"image_{idx}_lon"
            
            try:
                img_lat = float(form.get(lat_key, 0.0))
                img_lon = float(form.get(lon_key, 0.0))
            except:
                img_lat = 0.0
                img_lon = 0.0
            
            image_lats.append(img_lat)
            image_lons.append(img_lon)
        
        # Extract other form fields
        latitude = float(form.get("latitude"))
        longitude = float(form.get("longitude"))
        shop_size_sqft = int(form.get("shop_size_sqft", 150))
        email = form.get("email")
        
        # Evaluate the loan using our model
        result = model_service.evaluate_loan(
            images_bytes=image_bytes_list, 
            lat=latitude, 
            lon=longitude,
            shop_size_sqft=shop_size_sqft,
            image_lats=image_lats if any(image_lats) else None,
            image_lons=image_lons if any(image_lons) else None,
            email=email
        )
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
