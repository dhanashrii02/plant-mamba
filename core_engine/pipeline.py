"""
Plant Diagnostic Pipeline
Unified end-to-end inference system for Plant Mamba:
1. Image Quality Assessment (sharpness, lighting)
2. Vision Mamba Disease Classification (prediction, confidence, top-k)
3. Explainable AI & Severity Estimation (Grad-CAM heatmap overlay, % infected tissue)
4. Agronomic Treatment Recommendations
"""

import json
import os
import torch
from torchvision import transforms
from PIL import Image

from vision_mamba_model import VisionMambaClassifier, PLANT_CLASSES, CLEAN_CLASS_NAMES
from quality_assessor import assess_image_quality
from severity_and_xai import generate_gradcam_and_severity

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDiagnosticPipeline:
    def __init__(self, checkpoint_path=None, treatments_path=None, device=None, img_size=176):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

        # Resolve checkpoint path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if checkpoint_path is None:
            default_ckpt = os.path.join(base_dir, 'checkpoint.pt')
            if os.path.exists(default_ckpt):
                checkpoint_path = default_ckpt

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = VisionMambaClassifier.load_trained(checkpoint_path, device=self.device)
            print(f"[Pipeline] Loaded model checkpoint from: {checkpoint_path}")
        else:
            print("[Pipeline] WARNING: Checkpoint not found. Initializing model with default weights.")
            self.model = VisionMambaClassifier(num_classes=len(PLANT_CLASSES)).to(self.device)
            self.model.eval()

        # Resolve treatments path
        if treatments_path is None:
            treatments_path = os.path.join(base_dir, 'treatment_recommendations.json')

        self.treatments = {}
        if os.path.exists(treatments_path):
            try:
                with open(treatments_path, 'r', encoding='utf-8') as f:
                    self.treatments = json.load(f)
                print(f"[Pipeline] Loaded treatment guidelines for {len(self.treatments)} classes.")
            except Exception as e:
                print(f"[Pipeline] Could not load treatments: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def diagnose(self, image_input):
        """
        Runs the complete diagnostic suite on an image.
        image_input: file path (str) or PIL.Image.Image
        """
        if isinstance(image_input, str):
            pil_image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert('RGB')
        else:
            raise ValueError(f"Expected file path or PIL Image, got {type(image_input)}")

        # Step 1: Quality & Botanical Specimen Validation
        quality = assess_image_quality(pil_image)
        if not quality.get("is_valid_leaf", True):
            return {
                "is_valid_leaf": False,
                "rejection_code": quality.get("rejection_code", "INVALID_SPECIMEN"),
                "quality": quality,
                "predicted_class": None,
                "clean_name": None,
                "confidence": 0.0,
                "top_predictions": [],
                "severity_level": None,
                "infected_ratio": 0.0,
                "cam_image": None,
                "treatment": None,
                "original_image": pil_image,
            }

        # Step 2: Preprocess and predict with Test-Time Augmentation (TTA)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        tensor_flip = torch.flip(tensor, dims=[3])  # Horizontal flip TTA
        
        with torch.no_grad():
            logits_1, _ = self.model(tensor)
            logits_2, _ = self.model(tensor_flip)
            avg_probs = torch.softmax((logits_1 + logits_2) / 2.0, dim=-1)[0]
        
        conf, idx = avg_probs.max(dim=-1)
        confidence = conf.item() * 100.0
        pred_class = PLANT_CLASSES[idx.item()]
        all_probs = avg_probs.tolist()
        top_scores, top_indices = torch.topk(avg_probs, k=min(5, len(PLANT_CLASSES)))
        top_k = [(PLANT_CLASSES[i.item()], top_scores[j].item() * 100.0) for j, i in enumerate(top_indices)]
        class_idx = idx.item()

        # Find best disease class among non-healthy categories
        disease_indices = [i for i, c in enumerate(PLANT_CLASSES) if "healthy" not in c.lower()]
        best_disease_idx = max(disease_indices, key=lambda idx: all_probs[idx]) if disease_indices else class_idx

        # Step 3: Explainability and severity
        xai_result = generate_gradcam_and_severity(
            model=self.model,
            input_tensor=tensor,
            pil_image=pil_image,
            target_class_idx=class_idx,
            disease_class_idx=best_disease_idx
        )

        infected_ratio = xai_result["infected_ratio"]
        severity_level = xai_result["severity_level"]

        # Clinical Consistency Guard:
        # If significant pathological lesions exist (>= 3.0%) but the whole-leaf classifier
        # was biased by surrounding green tissue into predicting 'healthy':
        # Re-align diagnosis to the top matching pathological condition!
        if infected_ratio >= 3.0 and "healthy" in pred_class.lower():
            disease_pred_class = PLANT_CLASSES[best_disease_idx]
            disease_sum = sum(all_probs[i] for i in disease_indices)
            rel_conf = (all_probs[best_disease_idx] / max(disease_sum, 1e-6)) * 100.0
            pred_class = disease_pred_class
            confidence = max(all_probs[best_disease_idx] * 100.0, rel_conf * 0.75, 58.5)
            top_k = [(disease_pred_class, confidence)] + [(c, p) for c, p in top_k if c != disease_pred_class][:4]

        # Step 4: Treatment recommendation
        clean_name = CLEAN_CLASS_NAMES.get(pred_class, pred_class)
        treatment_info = self.treatments.get(pred_class, self.treatments.get(clean_name, {
            "disease": clean_name,
            "crop": clean_name.split()[0],
            "type": "Healthy" if "healthy" in pred_class.lower() else "Unknown",
            "treatment": "No specific treatment recorded. Consult a local agricultural specialist.",
            "prevention": "Ensure good cultural sanitation and inspect crops regularly.",
            "severity_note": "Monitor field condition."
        }))

        return {
            "is_valid_leaf": True,
            "predicted_class": pred_class,
            "clean_name": clean_name,
            "confidence": round(confidence, 2),
            "top_predictions": [(cls, round(p, 2)) for cls, p in top_k],
            "quality": quality,
            "severity_level": severity_level,
            "infected_ratio": infected_ratio,
            "cam_image": xai_result["cam_image"],
            "treatment": treatment_info,
            "original_image": pil_image,
        }


if __name__ == '__main__':
    import sys
    print("Testing PlantDiagnosticPipeline initialization...")
    pipeline = PlantDiagnosticPipeline()
    print("Pipeline ready.")
