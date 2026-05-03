import io
from PIL import Image
import PIL.ExifTags
import math
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

train_data = pd.DataFrame({
    'sku_count': [450, 120, 800, 50, 300, 600, 150, 900],
    'shelf_density': [0.85, 0.40, 0.95, 0.20, 0.70, 0.88, 0.50, 0.98],
    'competitors_500m': [3, 1, 6, 0, 2, 4, 1, 5],
    'footfall_index': [8, 2, 9, 1, 5, 7, 3, 10],
    'sku_diversity': [7, 2, 9, 1, 4, 6, 2, 10],
    'shop_size_sqft': [200, 100, 400, 80, 150, 250, 120, 500]
})
y_sales = np.array([18500, 7500, 28000, 4500, 12000, 22000, 8000, 35000])

model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.1, verbose=-1, min_child_samples=1)
model_lower.fit(train_data, y_sales)

model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.9, verbose=-1, min_child_samples=1)
model_upper.fit(train_data, y_sales)

# ==========================================
# 2. Advanced Logic Functions
# ==========================================
def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula
    R = 6371.0 
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # Return in meters

def validate_image_locations(image_capture_lats, image_capture_lons, submitted_lat, submitted_lon, max_distance_threshold=50):
    """
    Validate that all captured images are from the same location
    Returns: (is_valid, location_variance_meters, flag_message)
    """
    if not image_capture_lats or not image_capture_lons:
        # No capture location metadata, validation skipped
        return True, 0, ""
    
    max_distance = 0
    location_mismatches = []
    
    # Check each image against submitted location
    for idx, (lat, lon) in enumerate(zip(image_capture_lats, image_capture_lons)):
        if lat == 0 and lon == 0:
            continue  # Skip if no GPS data
        
        distance = calculate_distance(submitted_lat, submitted_lon, lat, lon)
        max_distance = max(max_distance, distance)
        
        if distance > max_distance_threshold:
            location_mismatches.append((idx, distance))
    
    if location_mismatches:
        flag_msg = f"Location variance detected: Image(s) {[x[0] for x in location_mismatches]} taken {max(x[1] for x in location_mismatches):.0f}m away from submitted location."
        return False, max_distance, flag_msg
    
    return True, max_distance, ""

def extract_exif_gps(img):
    try:
        exif = img._getexif()
        if not exif: return None
        
        gps_info = {}
        for tag, value in exif.items():
            decoded = PIL.ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = PIL.ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = float(gps_info["GPSLatitude"][0]) + float(gps_info["GPSLatitude"][1])/60 + float(gps_info["GPSLatitude"][2])/3600
            if gps_info.get("GPSLatitudeRef") == "S": lat = -lat
            
            lon = float(gps_info["GPSLongitude"][0]) + float(gps_info["GPSLongitude"][1])/60 + float(gps_info["GPSLongitude"][2])/3600
            if gps_info.get("GPSLongitudeRef") == "W": lon = -lon
            
            return lat, lon
    except:
        pass
    return None

def get_geo_signals(lat, lon, radius=500):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query_comp = f'[out:json];(nwr["shop"="convenience"](around:{radius},{lat},{lon});nwr["shop"="supermarket"](around:{radius},{lat},{lon});nwr["shop"="kiosk"](around:{radius},{lat},{lon});nwr["shop"="general"](around:{radius},{lat},{lon});nwr["shop"="grocery"](around:{radius},{lat},{lon}););out count;'
    overpass_query_footfall = f'[out:json];(nwr["amenity"="school"](around:{radius},{lat},{lon});nwr["amenity"="hospital"](around:{radius},{lat},{lon});nwr["landuse"="residential"](around:{radius},{lat},{lon});nwr["building"="apartments"](around:{radius},{lat},{lon});nwr["office"](around:{radius},{lat},{lon}););out count;'
    headers = {'User-Agent': 'TenZorX_Underwriting_Engine_v1'}
    competitors = 3
    footfall_index = 5
    
    try:
        res_comp = requests.get(overpass_url, params={'data': overpass_query_comp}, headers=headers, timeout=5)
        if res_comp.status_code == 200: competitors = int(res_comp.json()['elements'][0]['tags']['total'])
            
        res_foot = requests.get(overpass_url, params={'data': overpass_query_footfall}, headers=headers, timeout=5)
        if res_foot.status_code == 200:
            raw_footfall = int(res_foot.json()['elements'][0]['tags']['total'])
            footfall_index = min(10, max(1, int(raw_footfall / 2)))
    except: pass
    return competitors, footfall_index

def calculate_shelf_density(boxes, img_width, img_height):
    if len(boxes) == 0: return 0.0, 10.0
    
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    min_x, min_y = img_width, img_height
    max_x, max_y = 0, 0
    sum_individual_areas = 0
    
    for box in boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        mask[y1:y2, x1:x2] = 1
        
        box_area = max(1, x2 - x1) * max(1, y2 - y1)
        sum_individual_areas += box_area
        
        if x1 < min_x: min_x = x1
        if y1 < min_y: min_y = y1
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2
        
    active_width = max(1, max_x - min_x)
    active_height = max(1, max_y - min_y)
    active_shelf_area = active_width * active_height
    
    union_area = np.sum(mask)
    density = union_area / active_shelf_area
    
    # Store Organization Proxy: If union area == sum of areas, there is NO overlap (perfectly organized).
    org_ratio = union_area / max(1, sum_individual_areas)
    organization_score = min(10.0, max(1.0, round(org_ratio * 10, 1)))
    
    return round(min(1.0, float(density)), 2), organization_score

def process_images(image_bytes_list, form_lat, form_lon):
    total_skus = 0
    total_density = 0.0
    total_org = 0.0
    densities = []
    brightness_scores = []
    unique_classes = set()
    num_images = len(image_bytes_list)
    is_gps_spoofed = False
    
    if num_images == 0: return 0, 0.0, 0, 0, False, False, False, 5.0
    
    for img_bytes in image_bytes_list:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        img_gps = extract_exif_gps(img)
        if img_gps and form_lat and form_lon:
            try:
                dist = calculate_distance(float(form_lat), float(form_lon), img_gps[0], img_gps[1])
                if dist > 0.5: is_gps_spoofed = True # Over 500 meters mismatch!
            except: pass
            
        img_np = np.array(img)
        img_height, img_width = img_np.shape[:2]
        brightness_scores.append(np.mean(img_np))
        
        results = yolo_model(img_np, imgsz=1024, conf=0.15, iou=0.45)
        boxes = results[0].boxes
        
        if boxes.cls is not None:
            unique_classes.update(boxes.cls.cpu().numpy().tolist())
            
        sku_count = len(boxes) * 15
        density, org_score = calculate_shelf_density(boxes, img_width, img_height)
        
        densities.append(density)
        total_skus += sku_count
        total_density += density
        total_org += org_score
        
    avg_sku = total_skus
    avg_density = round(total_density / num_images, 2)
    avg_org_score = round(total_org / num_images, 1)
    
    sku_diversity_score = min(10, len(unique_classes))
    inventory_value_estimate = avg_sku * 40 
    
    density_variance = np.var(densities) if len(densities) > 1 else 0.0
    is_biased_photography = bool(density_variance > 0.15) 
    
    is_low_light = False
    if len(brightness_scores) > 0 and np.mean(brightness_scores) < 60:
        is_low_light = True
    
    return avg_sku, avg_density, sku_diversity_score, inventory_value_estimate, is_biased_photography, is_low_light, is_gps_spoofed, avg_org_score

def evaluate_loan(images_bytes, lat, lon, shop_size_sqft=150, image_lats=None, image_lons=None, email=None):
    avg_sku, avg_density, sku_diversity, inventory_value, is_biased_photo, is_low_light, is_gps_spoofed, org_score = process_images(images_bytes, lat, lon)
    competitors, footfall_index = get_geo_signals(lat, lon)
    
    # Validate image capture locations
    location_fraud_flag = False
    location_variance_distance = 0
    if image_lats and image_lons:
        # Estimate shop radius in meters (1 sqft = ~0.0929 sqm, Area = pi * r^2)
        estimated_area_sqm = shop_size_sqft * 0.092903
        estimated_radius_m = math.sqrt(estimated_area_sqm / math.pi)
        # We allow the estimated radius + 15m buffer for standard GPS variance
        dynamic_threshold = estimated_radius_m + 15.0
        
        is_valid, variance_meters, flag_msg = validate_image_locations(image_lats, image_lons, lat, lon, max_distance_threshold=dynamic_threshold)
        if not is_valid:
            location_fraud_flag = True
            location_variance_distance = variance_meters
    
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
    
    max_affordable_emi = int(monthly_inc_lower * 0.30)
    total_pre_approved_loan_amount = max_affordable_emi * 20
    
    # Staged disbursement logic to avoid single-day stock stuffing fraud
    initial_sanction_amount = int(total_pre_approved_loan_amount * 0.20) # 20% upfront (approx 2 months runway)
    final_sanction_amount = total_pre_approved_loan_amount - initial_sanction_amount # 80% remaining unlocked after 6 months

    monitoring_schedule = {
        "description": "Continuous Monitoring to prevent inventory stuffing fraud.",
        "email_target": email if email else "not_provided@example.com",
        "starting_week": "3 random days of the week, 3 times a day.",
        "months_1_to_6": "Random days in 5-6 months, continuous 3 times a day alerts."
    }
    
    market_percentile = "Top 50%"
    if monthly_rev_upper > 600000 and footfall_index >= 5: market_percentile = "Top 15%"
    elif monthly_rev_upper > 900000: market_percentile = "Top 5%"
    elif monthly_rev_upper < 150000: market_percentile = "Bottom 25%"
    
    risk_flags = []
    if inventory_value > 100000 and footfall_index < 3: risk_flags.append("inventory_footfall_mismatch")
    if avg_density > 0.95: risk_flags.append("overstocked_possible_inspection_gaming")
    elif 0.60 <= avg_density <= 0.85: risk_flags.append("healthy_turnover_refill_signal")
    if shop_size_sqft > 300 and sku_diversity < 3: risk_flags.append("diversity_size_mismatch")
    if is_biased_photo: risk_flags.append("inconsistent_stock_distribution_possible_biased_photography")
    if is_low_light: risk_flags.append("low_light_conditions_detected_proceed_with_caution")
    if is_gps_spoofed: risk_flags.append("CRITICAL: geolocation_spoofing_detected_exif_mismatch")
    if location_fraud_flag: risk_flags.append(f"CRITICAL: images_taken_from_different_locations_max_variance_{location_variance_distance:.0f}m")
        
    penalty = 0.0
    for flag in risk_flags:
        if "mismatch" in flag or "gaming" in flag or "biased" in flag or "CRITICAL" in flag:
            penalty += 0.20
    confidence_score = round(max(0.1, confidence_score - penalty), 2)
    
    memo = f"Store evaluated in the {market_percentile} for its catchment area (Footfall Index: {footfall_index}/10). "
    memo += f"Visual management quality is {'excellent' if org_score >= 8 else 'moderate' if org_score >= 5 else 'poor'} with an Organization Score of {org_score}/10. "
    if len(risk_flags) == 0:
        memo += "No risk flags detected. Recommended for immediate maximum capital disbursement."
    else:
        memo += f"However, {len(risk_flags)} risk flags were triggered, including: '{risk_flags[0].replace('_', ' ')}'. Manual review recommended."
    
    return {
        "daily_sales_range": [daily_lower, daily_upper],
        "monthly_revenue_range": [monthly_rev_lower, monthly_rev_upper],
        "monthly_income_range": [monthly_inc_lower, monthly_inc_upper],
        "confidence_score": confidence_score,
        "risk_flags": risk_flags if risk_flags else ["none_detected"],
        "recommendation": "approve_tier_1" if confidence_score > 0.75 else "approve_tier_2" if confidence_score > 0.5 else "needs_verification",
        "loan_details": {
            "max_affordable_emi_inr": max_affordable_emi,
            "pre_approved_loan_amount_inr": total_pre_approved_loan_amount,
            "initial_sanction_amount_inr": initial_sanction_amount,
            "final_sanction_amount_inr": final_sanction_amount,
            "market_percentile": market_percentile,
            "underwriter_memo": memo,
            "monitoring_schedule": monitoring_schedule
        },
        "latent_variables": {
            "inventory_value_estimate_inr": inventory_value,
            "footfall_proxy_index": footfall_index,
            "sku_diversity_score": sku_diversity,
            "shelf_density_index": avg_density,
            "organization_score": org_score
        },
        "extracted_features": {
            "avg_sku": avg_sku,
            "avg_density": avg_density,
            "competitors": competitors
        }
    }
