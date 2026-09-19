"""
Quality Assessment & Universal Botanical Specimen Validation Engine
Validates whether an input image is a genuine botanical plant leaf specimen before running inference.
Detects and rejects smartphones, electronic devices, human portraits, faces, documents, animals,
vehicles, household furniture, and arbitrary non-plant objects.
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights

# Initialize cached open-domain object recognition model
_OBJECT_MODEL = None
_CATEGORIES = None
_TRANSFORM = None


def _get_object_model():
    global _OBJECT_MODEL, _CATEGORIES, _TRANSFORM
    if _OBJECT_MODEL is None:
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            _OBJECT_MODEL = models.mobilenet_v3_small(weights=weights).eval()
            _CATEGORIES = weights.meta['categories']
            _TRANSFORM = weights.transforms()
        except Exception as e:
            print(f"[QualityAssessor] Could not load open-domain object model: {e}")
    return _OBJECT_MODEL, _CATEGORIES, _TRANSFORM


def assess_image_quality(image: Image.Image) -> dict:
    """
    Universal Botanical Specimen & Quality Assessment:
    1. Open-Domain Object Recognition (detects smartphones, electronics, vehicles, documents, animals).
    2. Human Facial & Skin Analysis (YCbCr chrominance).
    3. Botanical Chlorophyll & Organic Foliage Physics.
    4. Optical Focus & Lighting (Laplacian variance & luminance).
    """
    if isinstance(image, np.ndarray):
        pil_rgb = Image.fromarray(image).convert("RGB")
    else:
        pil_rgb = image.convert("RGB")
    img_rgb = np.array(pil_rgb)
    h, w, _ = img_rgb.shape
    total_pixels = h * w

    # --- 1. Open-Domain Object Recognition ---
    model, categories, transform = _get_object_model()
    detected_object_en = None
    detected_object_hi = None
    detected_object_mr = None
    rejection_code = None
    is_electronic_or_phone = False
    is_animal = False
    is_document_or_paper = False
    is_vehicle = False
    is_general_non_plant = False

    # --- 2. Botanical Foliage & Chlorophyll Analysis ---
    r = img_rgb[:, :, 0].astype(float)
    g = img_rgb[:, :, 1].astype(float)
    b = img_rgb[:, :, 2].astype(float)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # True chlorophyll reflection: G > R and G > B
    green_foliage = (
        (g > r) & (g > b) &
        (hsv[:, :, 0] >= 26) & (hsv[:, :, 0] <= 90) &
        (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25)
    )

    # Pathological foliar lesions (brown / rust / yellow dried tissue)
    necrotic_lesions = (
        (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 32) &
        (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 25) &
        (r > b + 12)
    )

    plant_matter = green_foliage | necrotic_lesions
    plant_ratio = float(np.sum(plant_matter) / total_pixels)

    if model is not None:
        try:
            t = transform(pil_rgb).unsqueeze(0)
            with torch.no_grad():
                logits = model(t)
                probs = torch.softmax(logits, dim=-1)[0]
                top_prob, top_idx_t = probs.max(dim=-1)
                top_idx = top_idx_t.item()
                top_label = categories[top_idx]

            top_prob_pct = top_prob.item() * 100.0

            # Electronic devices & smartphones
            phone_keywords = ['phone', 'cellular', 'telephone', 'laptop', 'computer', 'ipod', 'screen', 'monitor', 'keyboard', 'mouse', 'television', 'radio', 'modem', 'printer']
            if any(k in top_label.lower() for k in phone_keywords) and top_prob_pct > 8.0:
                is_electronic_or_phone = True
                detected_object_en = f"Smartphone / Electronic Device ({top_label})"
                detected_object_hi = f"स्मार्टफोन / इलेक्ट्रॉनिक डिवाइस ({top_label})"
                detected_object_mr = f"स्मार्टफोन / इलेक्ट्रॉनिक डिव्हाइस ({top_label})"

            # Animals & Pets (Dogs 151-268, Cats/Felines 281-293, Bears/Farm Mammals 294-397)
            if (151 <= top_idx <= 293 or 330 <= top_idx <= 397) and top_prob_pct > 15.0:
                is_animal = True
                detected_object_en = f"Animal / Pet ({top_label})"
                detected_object_hi = f"पशु / पालतू जानवर ({top_label})"
                detected_object_mr = f"प्राणी / पाळीव प्राणी ({top_label})"
            elif (0 <= top_idx <= 150) and top_prob_pct > 25.0 and plant_ratio < 0.30:
                is_animal = True
                detected_object_en = f"Animal / Wildlife ({top_label})"
                detected_object_hi = f"पशु / वन्यजीव ({top_label})"
                detected_object_mr = f"प्राणी / वन्यजीव ({top_label})"

            # Documents, paper, marksheets
            doc_keywords = ['menu', 'envelope', 'packet', 'book', 'paper', 'web site', 'crossword', 'comic book', 'binder', 'carton', 'wallet', 'passport']
            if any(k in top_label.lower() for k in doc_keywords) and top_prob_pct > 10.0 and plant_ratio < 0.15:
                is_document_or_paper = True
                detected_object_en = f"Document / Paper ({top_label})"
                detected_object_hi = f"दस्तावेज़ / कागज़ ({top_label})"
                detected_object_mr = f"कागदपत्र / कागद ({top_label})"

            # Vehicles & Transport
            vehicle_keywords = ['car', 'automobile', 'truck', 'bus', 'van', 'jeep', 'trailer', 'tractor', 'bicycle', 'bike', 'motorcycle', 'scooter', 'moped', 'convertible', 'minivan', 'pickup', 'cab', 'racer', 'train', 'boat', 'ship', 'airplane']
            if any(k in top_label.lower() for k in vehicle_keywords) and top_prob_pct > 15.0 and plant_ratio < 0.20:
                is_vehicle = True
                detected_object_en = f"Vehicle / Transport ({top_label})"
                detected_object_hi = f"वाहन / गाड़ी ({top_label})"
                detected_object_mr = f"वाहन / गाडी ({top_label})"

            # Household items / Furniture / Clothing
            household_keywords = ['chair', 'desk', 'table', 'couch', 'sofa', 'wardrobe', 'bed', 'lamp', 'bottle', 'cup', 'mug', 'plate', 'bowl', 'vase', 'clock', 'bucket', 'pillow', 'refrigerator', 'microwave', 'toaster', 'oven', 'stove', 'sink', 'bathtub', 'toilet', 'shoe', 'boot', 'sandal', 'sock', 'backpack', 'umbrella', 'suit', 'coat', 'jacket', 'jersey', 'dress', 'shirt', 'jean', 'skirt', 'tie', 'hat', 'cap', 'helmet', 'sunglasses', 'glasses']
            if any(k in top_label.lower() for k in household_keywords) and top_prob_pct > 15.0 and plant_ratio < 0.20:
                is_general_non_plant = True
                detected_object_en = f"Household Object / Furniture ({top_label})"
                detected_object_hi = f"घरेलू वस्तु / फर्नीचर ({top_label})"
                detected_object_mr = f"घरगुती वस्तू / फर्निचर ({top_label})"
        except Exception:
            pass

    # --- 3. Human Skin & Portrait Analysis (YCbCr Chrominance) ---
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycbcr[:, :, 1]
    cb = ycbcr[:, :, 2]
    skin_pixels = (cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 135)
    total_skin_ratio = float(np.sum(skin_pixels) / total_pixels)

    ch_start, ch_end = int(h * 0.20), int(h * 0.70)
    cw_start, cw_end = int(w * 0.20), int(w * 0.80)
    center_area = (ch_end - ch_start) * (cw_end - cw_start)
    center_skin = skin_pixels[ch_start:ch_end, cw_start:cw_end]
    center_skin_ratio = float(np.sum(center_skin) / max(center_area, 1))

    # --- 4. Optical Sharpness & Illumination ---
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    # --- 5. Universal Decision Gate ---
    is_valid_leaf = True
    status = "Passed"

    is_human = (center_skin_ratio > 0.35 and plant_ratio < 0.25) or (total_skin_ratio > 0.22 and plant_ratio < 0.20)
    
    if is_human:
        is_valid_leaf = False
        status = "Rejected"
        rejection_code = "HUMAN_PORTRAIT"
        details_en = "Human face or portrait photo detected. This is not a plant leaf specimen."
        details_hi = "मानव चेहरा या व्यक्ति का फोटो पहचाना गया! यह पौधे की पत्ती नहीं है। कृपया केवल फसल या पौधे की पत्ती की तस्वीर अपलोड करें।"
        details_mr = "मानवी चेहरा किंवा व्यक्तीचा फोटो आढळला! हे वनस्पतीचे पान नाही. कृपया केवळ पिकाच्या किंवा झाडाच्या पानाचा फोटो अपलोड करा."
    elif is_electronic_or_phone and plant_ratio < 0.20:
        is_valid_leaf = False
        status = "Rejected"
        rejection_code = "SMARTPHONE_OR_ELECTRONICS"
        details_en = f"Smartphone or electronic device detected ({detected_object_en}). This is not a plant leaf specimen."
        details_hi = f"स्मार्टफोन / इलेक्ट्रॉनिक डिवाइस ({detected_object_hi}) पहचानी गई! यह पौधे की पत्ती नहीं है। कृपया केवल फसल या पौधे की पत्ती की तस्वीर अपलोड करें।"
        details_mr = f"स्मार्टफोन / इलेक्ट्रॉनिक डिव्हाइस ({detected_object_mr}) आढळले! हे वनस्पतीचे पान नाही. कृपया केवळ पिकाच्या किंवा झाडाच्या पानाचा फोटो अपलोड करा."
    elif is_animal:
        is_valid_leaf = False
        status = "Rejected"
        rejection_code = "ANIMAL_OR_PET"
        details_en = f"Animal, pet, or wildlife detected ({detected_object_en}). This is not a plant leaf specimen."
        details_hi = f"पशु या पालतू जानवर ({detected_object_hi}) पहचाना गया! यह पौधे की पत्ती नहीं है। कृपया केवल पौधे की पत्ती अपलोड करें।"
        details_mr = f"प्राणी किंवा पाळीव प्राणी आढळला ({detected_object_mr})! हे वनस्पतीचे पान नाही. कृपया केवळ पानाचा फोटो अपलोड करा."
    elif is_document_or_paper:
        is_valid_leaf = False
        status = "Rejected"
        rejection_code = "DOCUMENT_OR_PAPER"
        details_en = f"Paper document, card, or text graphic detected ({detected_object_en}). This is not a plant leaf specimen."
        details_hi = f"दस्तावेज़, कागज़ या ग्राफिक ({detected_object_hi}) पहचाना गया! यह पौधे की पत्ती नहीं है। कृपया केवल पौधे की पत्ती अपलोड करें।"
        details_mr = f"कागदपत्र किंवा मजकूर आढळला ({detected_object_mr})! हे वनस्पतीचे पान नाही. कृपया केवळ पानाचा फोटो अपलोड करा."
    elif is_vehicle:
        is_valid_leaf = False
        status = "Rejected"
        rejection_code = "VEHICLE"
        details_en = f"Vehicle or machinery detected ({detected_object_en}). This is not a plant leaf specimen."
        details_hi = f"वाहन या मशीन ({detected_object_hi}) पहचानी गई! यह पौधे की पत्ती नहीं है।"
        details_mr = f"वाहन किंवा यंत्र आढळले ({detected_object_mr})! हे वनस्पतीचे पान नाही."
    elif is_general_non_plant:
        is_valid_leaf = False
        status = "Rejected"
        rejection_code = "NON_BOTANICAL_OBJECT"
        details_en = f"Non-botanical household object detected ({detected_object_en}). This is not a plant leaf specimen."
        details_hi = f"गैर-पादप घरेलू वस्तु ({detected_object_hi}) पहचानी गई! यह पौधे की पत्ती नहीं है।"
        details_mr = f"घरगुती गैर-वनस्पती वस्तू ({detected_object_mr}) आढळली! हे वनस्पतीचे पान नाही."
    elif plant_ratio < 0.12:
        is_valid_leaf = False
        status = "Rejected"
        rejection_code = "NOT_A_LEAF"
        details_en = "No plant foliage, chlorophyll, or botanical leaf tissue detected in image."
        details_hi = "छवि में कोई वानस्पतिक पत्ती, हरितद्रव्य या फसल का ऊतक नहीं मिला! कृपया पौधे की पत्ती की तस्वीर अपलोड करें।"
        details_mr = "प्रतिमेमध्ये वनस्पतीचे पान, हरितद्रव्य किंवा पानाचा भाग आढळला नाही! कृपया पिकाच्या पानाचा फोटो अपलोड करा."
    else:
        # Optical quality feedback
        messages_en, messages_hi, messages_mr = [], [], []
        if lap_var < 60:
            status = "Warning"
            messages_en.append(f"Image might be blurry (Sharpness Index: {lap_var:.1f}/60.0).")
            messages_hi.append(f"छवि थोड़ी धुंधली हो सकती है (शार्पनेस: {lap_var:.1f})।")
            messages_mr.append(f"प्रतिमा अस्पष्ट असू शकते (तीव्रता: {lap_var:.1f}).")
        if mean_brightness < 35 or mean_brightness > 235:
            status = "Warning"
            messages_en.append(f"Suboptimal illumination (Brightness: {mean_brightness:.1f}/255).")
            messages_hi.append(f"प्रकाश बहुत कम या बहुत अधिक है ({mean_brightness:.1f}/255)।")
            messages_mr.append(f"प्रकाश खूप कमी किंवा खूप जास्त आहे ({mean_brightness:.1f}/255).")

        if not messages_en:
            details_en = "Image resolution, focus, and illumination meet clinical requirements."
            details_hi = "छवि का रिज़ॉल्यूशन, फोकस और रोशनी नैदानिक आवश्यकताओं के पूर्णतः अनुकूल है।"
            details_mr = "प्रतिमेचे रिझोल्यूशन, फोकस आणि प्रकाश नैदानिक निकषांनुसार अचूक आहे."
        else:
            details_en = " ".join(messages_en)
            details_hi = " ".join(messages_hi)
            details_mr = " ".join(messages_mr)

    return {
        "status": status,
        "is_valid_leaf": is_valid_leaf,
        "rejection_code": rejection_code,
        "sharpness": lap_var,
        "brightness": mean_brightness,
        "plant_ratio": round(plant_ratio * 100, 1),
        "skin_ratio": round(total_skin_ratio * 100, 1),
        "center_skin_ratio": round(center_skin_ratio * 100, 1),
        "detected_object_en": detected_object_en,
        "detected_object_hi": detected_object_hi,
        "detected_object_mr": detected_object_mr,
        "details": details_en,
        "details_hi": details_hi,
        "details_mr": details_mr,
    }
