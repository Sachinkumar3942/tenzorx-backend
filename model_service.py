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

# Initialize baseline training data
train_data = pd.DataFrame({
    'sku_count': [450, 120, 800, 50, 300, 600, 150, 900],
    'shelf_density': [0.85, 0.40, 0.95, 0.20, 0.70, 0.88, 0.50, 0.98],
    'competitors_500m': [3, 1, 6, 0, 2, 4, 1, 5],
    'footfall_index': [8, 2, 9, 1, 5, 7, 3, 10],
    'sku_diversity': [7, 2, 9, 1, 4, 6, 2, 10],
    'shop_size_sqft': [200, 100, 400, 80, 150, 250, 120, 500]
})
# Daily sales estimates (INR) mapped to store profiles
y_sales = np.array([18500, 7500, 28000, 4500, 12000, 22000, 8000, 35000])

model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.1, verbose=-1, min_child_samples=1)
model_lower.fit(train_data, y_sales)

model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.9, verbose=-1, min_child_samples=1)
model_upper.fit(train_data, y_sales)

# ==========================================
# 2. Logic Functions
# ==========================================
def get_geo_signals(lat, lon, radius=500):
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    overpass_query_comp = f"""
    [out:json];
    (
      nwr["shop"="convenience"](around:{radius},{lat},{lon});
      nwr["shop"="supermarket"](around:{radius},{lat},{lon});
      nwr["shop"="kiosk"](around:{radius},{lat},{lon});
      nwr["shop"="general"](around:{radius},{lat},{lon});
      nwr["shop"="grocery"](around:{radius},{lat},{lon});
    );
    out count;
    """
    
    overpass_query_footfall = f"""
    [out:json];
    (
      nwr["amenity"="school"](around:{radius},{lat},{lon});
      nwr["amenity"="hospital"](around:{radius},{lat},{lon});
      nwr["landuse"="residential"](around:{radius},{lat},{lon});
      nwr["building"="apartments"](around:{radius},{lat},{lon});
      nwr["office"](around:{radius},{lat},{lon});
    );
    out count;
    """
    
    headers = {'User-Agent': 'TenZorX_Underwriting_Engine_v1'}
    competitors = 3
    footfall_index = 5
    
    try:
        res_comp = requests.get(overpass_url, params={'data': overpass_query_comp}, headers=headers, timeout=5)
        if res_comp.status_code == 200:
            competitors = int(res_comp.json()['elements'][0]['tags']['total'])
            
        res_foot = requests.get(overpass_url, params={'data': overpass_query_footfall}, headers=headers, timeout=5)
        if res_foot.status_code == 200:
            raw_footfall = int(res_foot.json()['elements'][0]['tags']['total'])
            footfall_index = min(10, max(1, int(raw_footfall / 2)))
    except Exception as e:
        print(f"Error fetching geo signals: {e}")
        
    return competitors, footfall_index

def calculate_shelf_density(boxes, img_width, img_height):
    if len(boxes) == 0: return 0.0
    
    # Create a mask to calculate union of all bounding boxes to avoid double counting overlaps
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Track the outermost boundaries to estimate the "Active Shelf Area"
    min_x, min_y = img_width, img_height
    max_x, max_y = 0, 0
    
    for box in boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        mask[y1:y2, x1:x2] = 1
        
        if x1 < min_x: min_x = x1
        if y1 < min_y: min_y = y1
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2
        
    # Calculate the area of the shelf that actually contains products
    active_width = max(1, max_x - min_x)
    active_height = max(1, max_y - min_y)
    active_shelf_area = active_width * active_height
    
    # Density is the product area divided by the active shelf area, NOT the entire image
    # (which includes irrelevant floors and ceilings)
    product_area = np.sum(mask)
    density = product_area / active_shelf_area
    
    return round(min(1.0, float(density)), 2)

def process_images(image_bytes_list):
    total_skus = 0
    total_density = 0.0
    unique_classes = set()
    num_images = len(image_bytes_list)
    
    if num_images == 0:
        return 0, 0.0, 0, 0
    
    for img_bytes in image_bytes_list:
        # Load image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        img_height, img_width = img_np.shape[:2]
        
        # Run YOLO
        results = yolo_model(img_np, imgsz=1024, conf=0.15, iou=0.45)
        boxes = results[0].boxes
        
        # Extract unique classes for Diversity Score
        if boxes.cls is not None:
            unique_classes.update(boxes.cls.cpu().numpy().tolist())
            
        # Apply an expansion multiplier to account for non-standard retail packaging
        # not captured by the base object detection weights.
        sku_count = len(boxes) * 15
        density = calculate_shelf_density(boxes, img_width, img_height)
        
        total_skus += sku_count
        total_density += density
        
    # Summing the SKU counts across multiple shelf photos to estimate total store inventory.
    avg_sku = total_skus
    avg_density = round(total_density / num_images, 2)
    
    sku_diversity_score = min(10, len(unique_classes))
    inventory_value_estimate = avg_sku * 40 # Approx Rs 40 average per FMCG item
    
    return avg_sku, avg_density, sku_diversity_score, inventory_value_estimate

def evaluate_loan(images_bytes, lat, lon, shop_size_sqft=150):
    avg_sku, avg_density, sku_diversity, inventory_value = process_images(images_bytes)
    competitors, footfall_index = get_geo_signals(lat, lon)
    
    current_store = pd.DataFrame({
        'sku_count': [avg_sku],
        'shelf_density': [avg_density],
        'competitors_500m': [competitors],
        'footfall_index': [footfall_index],
        'sku_diversity': [sku_diversity],
        'shop_size_sqft': [shop_size_sqft]
    })
    
    daily_lower = int(model_lower.predict(current_store)[0])
    daily_upper = int(model_upper.predict(current_store)[0])
    
    # Enforce logical bounds on predictions
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
    
    # ---------------------------------------------------------
    # Fraud & Adversarial Logic Validation
    # ---------------------------------------------------------
    risk_flags = []
    
    if inventory_value > 100000 and footfall_index < 3:
        risk_flags.append("inventory_footfall_mismatch")
        
    if avg_density > 0.95:
        risk_flags.append("overstocked_possible_inspection_gaming")
    elif 0.60 <= avg_density <= 0.85:
        risk_flags.append("healthy_turnover_refill_signal")
        
    if shop_size_sqft > 300 and sku_diversity < 3:
        risk_flags.append("diversity_size_mismatch")
        
    # Penalize confidence if there are negative risk flags
    penalty = 0.0
    for flag in risk_flags:
        if "mismatch" in flag or "gaming" in flag:
            penalty += 0.15
    confidence_score = round(max(0.1, confidence_score - penalty), 2)
    
    return {
        "daily_sales_range": [daily_lower, daily_upper],
        "monthly_revenue_range": [monthly_rev_lower, monthly_rev_upper],
        "monthly_income_range": [monthly_inc_lower, monthly_inc_upper],
        "confidence_score": confidence_score,
        "risk_flags": risk_flags if risk_flags else ["none_detected"],
        "recommendation": "approve_tier_1" if confidence_score > 0.75 else "approve_tier_2" if confidence_score > 0.5 else "needs_verification",
        "latent_variables": {
            "inventory_value_estimate_inr": inventory_value,
            "footfall_proxy_index": footfall_index,
            "sku_diversity_score": sku_diversity,
            "shelf_density_index": avg_density
        },
        "extracted_features": {
            "avg_sku": avg_sku,
            "avg_density": avg_density,
            "competitors": competitors
        }
    }
