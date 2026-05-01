import io
from PIL import Image
import numpy as np
from ultralytics import YOLO
import requests
import lightgbm as lgb
import pandas as pd
import json

# ==========================================
# 1. Initialize Models on Startup
# ==========================================
yolo_model = YOLO("yolov10x.pt")

# Train mock regressors
train_data = pd.DataFrame({
    'sku_count': [450, 120, 800, 50, 300, 600, 150, 900],
    'shelf_density': [0.85, 0.40, 0.95, 0.20, 0.70, 0.88, 0.50, 0.98],
    'competitors_500m': [3, 1, 6, 0, 2, 4, 1, 5],
    'shop_size_sqft': [200, 100, 400, 80, 150, 250, 120, 500]
})
y_sales = np.array([8500, 2500, 14000, 900, 5000, 11000, 3000, 18000])

model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.1, verbose=-1)
model_lower.fit(train_data, y_sales)

model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.9, verbose=-1)
model_upper.fit(train_data, y_sales)

# ==========================================
# 2. Logic Functions
# ==========================================
def get_competitor_density(lat, lon, radius=500):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      nwr["shop"="convenience"](around:{radius},{lat},{lon});
      nwr["shop"="supermarket"](around:{radius},{lat},{lon});
    );
    out count;
    """
    headers = {'User-Agent': 'Kirana_Underwriting_Hackathon_Script_v1'}
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return int(data['elements'][0]['tags']['total'])
    except Exception as e:
        print(f"Error fetching competitors: {e}")
    return 3

def calculate_shelf_density(boxes, img_width, img_height):
    if len(boxes) == 0: return 0.0
    
    # Create a mask to calculate union of all bounding boxes to avoid double counting overlaps
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for box in boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        mask[y1:y2, x1:x2] = 1
        
    density = np.sum(mask) / (img_width * img_height)
    return round(float(density), 2)

def process_images(image_bytes_list):
    total_skus = 0
    total_density = 0.0
    num_images = len(image_bytes_list)
    
    if num_images == 0:
        return 0, 0.0
    
    for img_bytes in image_bytes_list:
        # Load image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        img_height, img_width = img_np.shape[:2]
        
        # Run YOLO
        results = yolo_model(img_np, imgsz=1024, conf=0.15, iou=0.45)
        boxes = results[0].boxes
        
        sku_count = len(boxes)
        density = calculate_shelf_density(boxes, img_width, img_height)
        
        total_skus += sku_count
        total_density += density
        
    avg_sku = int(total_skus / num_images)
    avg_density = round(total_density / num_images, 2)
    
    return avg_sku, avg_density

def evaluate_loan(images_bytes, lat, lon, shop_size_sqft=150):
    avg_sku, avg_density = process_images(images_bytes)
    competitors = get_competitor_density(lat, lon)
    
    current_store = pd.DataFrame({
        'sku_count': [avg_sku],
        'shelf_density': [avg_density],
        'competitors_500m': [competitors],
        'shop_size_sqft': [shop_size_sqft]
    })
    
    daily_lower = int(model_lower.predict(current_store)[0])
    daily_upper = int(model_upper.predict(current_store)[0])
    
    # Failsafe logic if predictions are wonky
    if daily_lower < 0: daily_lower = 0
    if daily_upper < daily_lower: daily_upper = daily_lower + 1000
    
    operating_days = 30
    margin_assumptions = [0.12, 0.15]
    monthly_rev_lower = daily_lower * operating_days
    monthly_rev_upper = daily_upper * operating_days
    monthly_inc_lower = int(monthly_rev_lower * margin_assumptions[0])
    monthly_inc_upper = int(monthly_rev_upper * margin_assumptions[1])
    
    spread_ratio = (daily_upper - daily_lower) / daily_upper if daily_upper > 0 else 0
    confidence_score = round(max(0.0, 1.0 - spread_ratio), 2)
    
    risk_flags = []
    if avg_sku > 500 and competitors < 1:
        risk_flags.append("inventory_footfall_mismatch")
    if avg_density > 0.90:
        risk_flags.append("overstocked_possible_inspection_gaming")
        
    return {
        "daily_sales_range": [daily_lower, daily_upper],
        "monthly_revenue_range": [monthly_rev_lower, monthly_rev_upper],
        "monthly_income_range": [monthly_inc_lower, monthly_inc_upper],
        "confidence_score": confidence_score,
        "risk_flags": risk_flags if risk_flags else ["none_detected"],
        "recommendation": "approve_tier_2" if confidence_score > 0.6 else "needs_verification",
        "extracted_features": {
            "avg_sku": avg_sku,
            "avg_density": avg_density,
            "competitors": competitors
        }
    }
