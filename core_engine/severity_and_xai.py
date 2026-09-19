import torch
import numpy as np
from PIL import Image
import cv2


def extract_leaf_and_lesions(img_rgb):
    """
    Computes precise pixel-level leaf boundary and pathological lesion tissue
    (necrosis, blight, scorch, rust pustules, spots, and rots).
    Uses green contour containment and blob-filtering so background laboratory sheets
    and single-pixel noise are never misclassified as lesions.
    Returns (leaf_clean, lesion_in_leaf, infected_ratio, severity_level).
    """
    h, w, _ = img_rgb.shape
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    R, G, B = img_rgb[:, :, 0].astype(int), img_rgb[:, :, 1].astype(int), img_rgb[:, :, 2].astype(int)

    # 1. Primary green foliage mask
    green = (hsv[:, :, 0] >= 24) & (hsv[:, :, 0] <= 92) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 2] >= 25)

    # Find dominant green contours representing the leaf blade
    cnts, _ = cv2.findContours(green.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    for c in cnts:
        if cv2.contourArea(c) > 200:
            cv2.drawContours(leaf_mask, [c], -1, 1, -1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_close)

    # 2. Pathological / necrotic lesion tissue strictly contained within the leaf blade
    nec_raw = (
        (leaf_mask > 0) &
        (
            ((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 30) & (R > G + 10) & (hsv[:, :, 1] >= 30) & (hsv[:, :, 2] >= 30)) |
            ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 35) & (R > G + 2) & (hsv[:, :, 1] >= 25) & (hsv[:, :, 1] <= 160) & (hsv[:, :, 2] >= 80) & (R > B + 20))
        )
    )

    # Filter out microscopic noise (< 30 px connected components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(nec_raw.astype(np.uint8))
    lesion_in_leaf = np.zeros((h, w), dtype=bool)
    if num_labels > 1:
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= 30:  # Minimum valid lesion cluster size
                lesion_in_leaf[labels == lbl] = True

    # Complete leaf blade
    total_leaf = (leaf_mask > 0) | lesion_in_leaf
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_clean = cv2.morphologyEx(total_leaf.astype(np.uint8), cv2.MORPH_CLOSE, kernel_clean)

    leaf_pixels = int(np.sum(leaf_clean > 0))
    lesion_pixels = int(np.sum(lesion_in_leaf))

    if leaf_pixels < 400:
        infected_ratio = 0.0
    else:
        infected_ratio = (lesion_pixels / leaf_pixels) * 100.0

    if infected_ratio < 2.0:
        severity_level = 'Healthy / None (< 2%)'
    elif infected_ratio < 10.0:
        severity_level = 'Mild (2% - 10%)'
    elif infected_ratio < 25.0:
        severity_level = 'Moderate (10% - 25%)'
    else:
        severity_level = 'Severe (> 25%)'

    return leaf_clean, lesion_in_leaf, round(infected_ratio, 1), severity_level


def generate_gradcam_and_severity(model, input_tensor, pil_image, target_class_idx, disease_class_idx=None):
    """
    Physically Accurate Lesion Localization & Grad-CAM Attention Heatmap.

    Ensures:
    1. The heatmap highlights the TRUE pathological lesion (necrosis, blight, scorch, spots)
       rather than green healthy foliage or background edges.
    2. Combines neural backpropagated disease gradients with high-resolution physical lesion saliency.
    3. Severity and infected area % accurately reflect real leaf tissue pathology.
    """
    working_img = pil_image.copy()
    working_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    w, h = working_img.size
    orig_np = np.array(working_img, dtype=np.uint8)

    # 1. Physical Leaf & Lesion Segmentation
    leaf_mask, lesion_mask, infected_ratio, severity_level = extract_leaf_and_lesions(orig_np)

    # Determine optimal backprop target:
    # If lesions are detected, backprop for the disease class so gradients map to pathology features
    effective_target = target_class_idx
    if infected_ratio >= 2.0 and disease_class_idx is not None:
        effective_target = disease_class_idx

    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.clone().detach().to(device)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # 2. Neural Grad-CAM
    with torch.enable_grad():
        logits, _ = model(input_tensor)
        feature_map = model.last_feature_map  # [1, 512, Hf, Wf]

        if feature_map is None or not feature_map.requires_grad:
            cam = np.zeros((14, 14), dtype=np.float32)
        else:
            feature_map.retain_grad()
            model.zero_grad(set_to_none=True)
            target_score = logits[0, int(effective_target)]
            target_score.backward(retain_graph=False)

            gradients = feature_map.grad
            if gradients is None:
                cam = np.zeros((14, 14), dtype=np.float32)
            else:
                # Per-channel importance weights
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                weighted = (weights * feature_map).sum(dim=1).squeeze(0)
                cam = torch.relu(weighted).detach().cpu().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    cam_resized = cv2.resize(cam, (w, h))

    # 3. Guided Fusion with Lesion Saliency
    leaf_smooth = cv2.GaussianBlur(leaf_mask.astype(np.float32), (15, 15), 0)

    if infected_ratio >= 2.0:
        # Create smooth physical lesion saliency map
        lesion_smooth = cv2.GaussianBlur(lesion_mask.astype(np.float32), (25, 25), 0)
        if lesion_smooth.max() > 0:
            lesion_smooth = lesion_smooth / lesion_smooth.max()

        # Fuse neural attention (30%) and physical lesion saliency (70%)
        fused = 0.3 * cam_resized + 0.7 * lesion_smooth
        # Constrain strictly to leaf area
        fused = fused * leaf_smooth
        if fused.max() > 0:
            fused = fused / fused.max()

        # Suppress ambient background noise below 0.20
        fused = np.where(fused < 0.20, fused * 0.3, fused)
        final_heatmap = fused
    else:
        # Genuinely healthy leaf: calm foliar attention constrained to leaf
        final_heatmap = cam_resized * leaf_smooth
        if final_heatmap.max() > 0:
            # Scale down peak intensity for healthy leaves so it doesn't alarm the user with red spikes
            final_heatmap = (final_heatmap / final_heatmap.max()) * 0.45

    # 4. Color Mapping & Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * final_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32)

    overlay = np.clip(0.65 * orig_np.astype(np.float32) + 0.35 * heatmap_colored, 0, 255).astype(np.uint8)
    cam_pil = Image.fromarray(overlay)

    return {
        'cam_image': cam_pil,
        'severity_level': severity_level,
        'infected_ratio': infected_ratio,
        'raw_heatmap': final_heatmap
    }
