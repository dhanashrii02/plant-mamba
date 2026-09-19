import os
import glob
import sys

sys.path.append(r"d:\plant_mamba_project\core_engine")
from pipeline import PlantDiagnosticPipeline
from vision_mamba_model import CLEAN_CLASS_NAMES

pipeline = PlantDiagnosticPipeline()
val_dir = r"d:\plant_mamba_project\core_engine\plantvillage_data\data_38\val"

test_classes = [
    "Apple___Cedar_apple_rust",
    "Corn_(maize)___Common_rust_",
    "Grape___Black_rot",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Potato___Late_blight",
    "Squash___Powdery_mildew",
    "Tomato___Early_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

print("\n=== MULTI-CROP 38-CLASS LIVE DIAGNOSIS TEST ===")
correct = 0
for raw_name in test_classes:
    folder = os.path.join(val_dir, raw_name)
    imgs = glob.glob(os.path.join(folder, "*.*"))
    if not imgs:
        print(f"[-] Missing images for {raw_name}")
        continue
    sample_img = imgs[0]
    res = pipeline.diagnose(sample_img)
    clean_gt = CLEAN_CLASS_NAMES.get(raw_name, raw_name)
    clean_pred = res["clean_name"]
    is_match = (res["predicted_class"] == raw_name)
    if is_match:
        correct += 1
    mark = "[PASS]" if is_match else "[DIFF]"
    print(f"{mark} Target: [{clean_gt}] -> Predicted: [{clean_pred}] (Confidence: {res['confidence']}%)")

print(f"\nFinal Score: {correct}/{len(test_classes)} ({correct/len(test_classes)*100:.1f}%)")
