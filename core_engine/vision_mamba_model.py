import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

PLANT_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

CLEAN_CLASS_NAMES = {
    'Apple___Apple_scab': 'Apple Scab',
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple___Cedar_apple_rust': 'Apple Cedar Rust',
    'Apple___healthy': 'Apple (Healthy)',
    'Blueberry___healthy': 'Blueberry (Healthy)',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry Powdery Mildew',
    'Cherry_(including_sour)___healthy': 'Cherry (Healthy)',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Corn Gray Leaf Spot',
    'Corn_(maize)___Common_rust_': 'Corn Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Corn Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Corn (Healthy)',
    'Grape___Black_rot': 'Grape Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape Black Measles (Esca)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight',
    'Grape___healthy': 'Grape (Healthy)',
    'Orange___Haunglongbing_(Citrus_greening)': 'Orange Citrus Greening (Huanglongbing)',
    'Peach___Bacterial_spot': 'Peach Bacterial Spot',
    'Peach___healthy': 'Peach (Healthy)',
    'Pepper,_bell___Bacterial_spot': 'Pepper Bell Bacterial Spot',
    'Pepper,_bell___healthy': 'Pepper Bell (Healthy)',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Potato (Healthy)',
    'Raspberry___healthy': 'Raspberry (Healthy)',
    'Soybean___healthy': 'Soybean (Healthy)',
    'Squash___Powdery_mildew': 'Squash Powdery Mildew',
    'Strawberry___Leaf_scorch': 'Strawberry Leaf Scorch',
    'Strawberry___healthy': 'Strawberry (Healthy)',
    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Spider Mites',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato___healthy': 'Tomato (Healthy)',
}


class BiDirectionalSSM(nn.Module):
    """Bidirectional State Space Block (Vision Mamba core).

    SPEED CHANGE: the original ran a plain Python `for t in range(N)` loop
    over every token, sequentially, for both the forward and backward pass.
    On a GPU this is slow because each step is a separate tiny kernel launch
    (Python loop overhead dominates, not the actual math). Because the decay
    gate here (`sigmoid(A.mean(dim=0))`) doesn't actually depend on t or on
    the input — it's the same constant vector every step — the whole
    recurrence has a closed-form solution:

        h_t = a*h_{t-1} + b_t   =>   h_t = a^t * cumsum(b_k * a^-k)

    This lets the entire scan run as a few vectorized tensor ops instead of
    N sequential Python steps. The gate is clamped to [0.2, 0.98] so the
    a^-k term stays numerically safe for the token counts used here (see
    VisionMambaClassifier's token pooling below).
    """

    def __init__(self, d_model=512, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_fwd = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.A_bwd = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.act = nn.SiLU()

    @staticmethod
    def _scan(b_proj, gate):
        """Vectorized linear recurrence h_t = gate*h_{t-1} + b_t.

        b_proj: [B, N, d_state]   gate: [d_state], constant across t.
        Returns h: [B, N, d_state].
        """
        B, N, S = b_proj.shape
        t = torch.arange(N, device=b_proj.device, dtype=b_proj.dtype)
        log_gate = torch.log(gate).unsqueeze(0)          # [1, S]
        log_pow = t.unsqueeze(-1) * log_gate              # [N, S]  == log(gate^t)
        # h_t = gate^t * cumsum_{k<=t}(b_k * gate^-k)
        scaled_in = b_proj * torch.exp(-log_pow).unsqueeze(0)   # [B, N, S]
        h = torch.cumsum(scaled_in, dim=1) * torch.exp(log_pow).unsqueeze(0)
        return h

    def forward(self, x):
        B, N, D = x.shape
        u = self.act(x)
        b_proj = self.B(u)  # [B, N, d_state]

        gate_fwd = torch.sigmoid(self.A_fwd.mean(dim=0)).clamp(0.2, 0.98)
        gate_bwd = torch.sigmoid(self.A_bwd.mean(dim=0)).clamp(0.2, 0.98)

        h_fwd = self._scan(b_proj, gate_fwd)                       # [B, N, d_state]
        h_bwd = self._scan(b_proj.flip(1), gate_bwd).flip(1)       # backward pass

        y_fwd = self.C(h_fwd) + self.D * u
        y_bwd = self.C(h_bwd) + self.D * u

        return self.out_proj(torch.cat([y_fwd, y_bwd], dim=-1))


class VisionMambaClassifier(nn.Module):
    """
    Agricultural Vision Mamba Classifier.

    CHANGES FROM THE ORIGINAL:
    1. Removed the random `torch.randn(...)` "prototypes" — they were never
       trained and carried zero disease-specific information.
    2. Removed the hard-coded RGB-color / contrast heuristic block that was
       overriding the model's output.
    3. Replaced the cosine-similarity-to-random-noise trick with a normal
       trainable linear classifier head, trainable with cross-entropy loss.
    4. `last_feature_map` is kept for the real Grad-CAM in severity_and_xai.py.
    5. SPEED: added `token_grid` pooling — the ResNet-18 feature map is
       downsampled (e.g. 7x7 -> 4x4) before being fed to the SSM, so there
       are fewer tokens to process. Combined with the vectorized scan above,
       this removes the training loop's biggest bottleneck.
    """

    def __init__(self, num_classes=len(PLANT_CLASSES), pretrained_backbone=True, token_grid=4):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.token_pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

        self.ssm = BiDirectionalSSM(d_model=512)
        self.norm = nn.LayerNorm(512)

        self.classes = PLANT_CLASSES
        self.last_feature_map = None

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feats = self.backbone(x)              # [B, 512, H, W]  (H=W=7 for 224px input)
        self.last_feature_map = feats          # kept for Grad-CAM (full-res, gradients preserved)

        pooled_feats = self.token_pool(feats)  # [B, 512, token_grid, token_grid] — fewer tokens for the SSM
        tokens = pooled_feats.flatten(2).transpose(1, 2)   # [B, token_grid*token_grid, 512]

        mamba_out = self.ssm(tokens)
        pooled = self.norm(mamba_out.mean(dim=1))   # [B, 512]

        logits = self.classifier(pooled)
        return logits, pooled

    def predict(self, x, top_k=3):
        """Convenience method: returns (predicted_class_name, confidence_%, all_probs, top_k_list)."""
        self.eval()
        if x.ndim == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)[0]
        
        conf, idx = probs.max(dim=-1)
        top_k = min(top_k, len(self.classes))
        top_scores, top_indices = torch.topk(probs, k=top_k)
        top_list = [
            (self.classes[i.item()], top_scores[j].item() * 100.0)
            for j, i in enumerate(top_indices)
        ]
        return self.classes[idx.item()], conf.item() * 100.0, probs.tolist(), top_list

    @classmethod
    def load_trained(cls, checkpoint_path, device=None, token_grid=4):
        """Loads a trained checkpoint safely whether it is a state_dict or a dict payload."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cls(num_classes=len(PLANT_CLASSES), pretrained_backbone=False, token_grid=token_grid)
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError(f"Unknown checkpoint format at {checkpoint_path}")
        model.to(device)
        model.eval()
        return model

