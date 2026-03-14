import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import os
import gdown
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GDRIVE_FILE_ID = "1Sf4U95DCqx1E8_BIicCie6V9aaCYY7nK"
MODEL_PATH     = "hybrid_rv_final.pth"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_model() -> None:
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model — first time only, please wait…"):
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                    MODEL_PATH, quiet=False,
                )
                st.success("✅ Model downloaded!")
            except Exception as exc:
                st.error(f"❌ Download failed: {exc}")
                st.info("Make sure the Google Drive file is shared as 'Anyone with the link'.")
                st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        for p in resnet.parameters():
            p.requires_grad = False
        for p in resnet.layer4.parameters():
            p.requires_grad = True
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 2048, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, heads: int = 8, depth: int = 6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=2048, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)

    def forward(self, x):
        return self.encoder(x)


class ResNetViT(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone    = ResNetBackbone()
        self.patch_embed = PatchEmbedding()
        self.transformer = TransformerEncoder()
        self.norm        = nn.LayerNorm(768)
        self.dropout     = nn.Dropout(0.3)
        self.fc          = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
#  GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks: list = []

        layer4     = model.backbone.features[-1]
        last_block = list(layer4.children())[-1]
        target     = last_block.conv3

        self._hooks.append(target.register_forward_hook(self._save_activation))
        self._hooks.append(target.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, _m, _i, out):
        self.activations = out.detach()

    def _save_gradient(self, _m, _gi, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        img_tensor = img_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            output = self.model(img_tensor)
            self.model.zero_grad()
            output[0, class_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224))

        weights = F.relu(self.gradients).mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode="bicubic", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()

        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            return np.zeros((224, 224))

        return np.power(cam, 1.8)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()


def apply_colormap(orig: Image.Image, cam: np.ndarray):
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    orig_arr = np.array(orig.resize((224, 224))).astype(np.float32)
    overlay  = np.clip(0.55 * orig_arr + 0.45 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(heatmap)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_logo_b64(path: str) -> str | None:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext  = path.rsplit(".", 1)[-1].lower()
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return f"data:{mime};base64,{data}"
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model      = ResNetViT(num_classes=2)
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model, device


def predict_single(model: nn.Module, device: torch.device, image: Image.Image) -> dict:
    img_rgb    = image.convert("RGB")
    img_tensor = TRANSFORM(img_rgb).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs        = model(img_tensor)
        probs          = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        class_idx      = predicted.item()

    gcam    = GradCAM(model)
    cam_map = gcam.generate(img_tensor.clone(), class_idx)
    gcam.remove_hooks()

    overlay, heatmap = apply_colormap(img_rgb, cam_map)

    confidence_pct = float(conf.item() * 100)
    if class_idx == 0:
        risk = "Low"
    else:
        risk = "High" if confidence_pct >= 85 else "Moderate"

    return {
        "prediction":     ["Normal", "Parkinson's Disease"][class_idx],
        "class_idx":      class_idx,
        "confidence":     confidence_pct,
        "normal_prob":    float(probs[0][0].item() * 100),
        "parkinson_prob": float(probs[0][1].item() * 100),
        "risk_level":     risk,
        "cam_overlay":    overlay,
        "cam_heatmap":    heatmap,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image":          img_rgb,
    }


def build_pdf(patient: dict, result: dict) -> bytes:
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    story  = []
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle("T", parent=styles["Heading1"], fontSize=20,
        textColor=colors.HexColor("#0d2b5e"), spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#3a7bd5"), spaceAfter=20,
        alignment=TA_CENTER)
    head_s  = ParagraphStyle("H", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#0d2b5e"), spaceAfter=8,
        spaceBefore=14, fontName="Helvetica-Bold")
    body_s  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
        textColor=colors.HexColor("#1a1a2e"), leading=14)
    warn_s  = ParagraphStyle("W", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#7f1d1d"), leading=13,
        backColor=colors.HexColor("#fff1f1"), borderPad=6)

    story.append(Paragraph("NeuroScan AI", title_s))
    story.append(Paragraph("Parkinson's Disease MRI Analysis Report", sub_s))

    def tbl(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#0d2b5e")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 9),
            ("TOPPADDING",    (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#f0f6ff"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 10),
            ("TOPPADDING",    (0, 1), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]))
        return t

    story.append(Paragraph("Patient Information", head_s))
    story.append(tbl([
        ["Field", "Details"],
        ["Patient Name",    patient["name"]],
        ["Patient ID",      patient["patient_id"]],
        ["Age",             str(patient["age"])],
        ["Gender",          patient["gender"]],
        ["Scan Date",       patient["scan_date"]],
        ["Referring Doctor",patient.get("doctor", "—")],
    ], [2*inch, 4.5*inch]))

    if patient.get("medical_history", "").strip():
        story.append(Spacer(1, 8))
        story.append(Paragraph("Medical History", head_s))
        story.append(Paragraph(patient["medical_history"], body_s))

    story.append(Paragraph("AI Analysis Results", head_s))
    story.append(tbl([
        ["Metric",                 "Value"],
        ["Diagnosis",              result["prediction"]],
        ["Confidence Score",       f"{result['confidence']:.2f}%"],
        ["Normal Probability",     f"{result['normal_prob']:.2f}%"],
        ["Parkinson's Probability",f"{result['parkinson_prob']:.2f}%"],
        ["Risk Level",             result["risk_level"]],
        ["Analysis Time",          result["timestamp"]],
        ["AI Model",               "ResNet50 + Vision Transformer (ViT)"],
    ], [2.5*inch, 4*inch]))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Brain MRI Scan & Grad-CAM Heatmap", head_s))
    img_buf = io.BytesIO(); result["image"].save(img_buf, "PNG"); img_buf.seek(0)
    cam_buf = io.BytesIO(); result["cam_overlay"].save(cam_buf, "PNG"); cam_buf.seek(0)
    img_tbl = Table([[
        RLImage(img_buf,  width=2.8*inch, height=2.8*inch),
        RLImage(cam_buf,  width=2.8*inch, height=2.8*inch),
    ]], colWidths=[3.2*inch, 3.2*inch])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("INNERGRID", (0, 0), (-1, -1), 0, colors.white),
        ("BOX",    (0, 0), (-1, -1), 0, colors.white),
    ]))
    story.append(img_tbl)
    caption_s = ParagraphStyle("C", parent=styles["Normal"], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER, spaceBefore=4)
    story.append(Paragraph("Left: Original MRI &nbsp;&nbsp;|&nbsp;&nbsp; Right: Grad-CAM Attention Overlay", caption_s))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI system for research and educational "
        "purposes only. It must NOT replace clinical diagnosis by a qualified medical professional. "
        "Always consult a licensed neurologist for any medical decisions.", warn_s))

    story.append(Spacer(1, 20))
    footer_s = ParagraphStyle("F", parent=styles["Normal"], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(
        f"NeuroScan AI · BVC College of Engineering, Palacharla · "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_s))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  ROYAL CSS — Deep Midnight Navy + Antique Gold + Playfair Display
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Royal Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,900;1,400;1,600&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Cinzel:wght@400;500;600;700;900&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

/* ── Royal Light Token Palette ── */
:root {
  --midnight:     #fdf8f0;
  --navy-deep:    #fff9f2;
  --navy:         #fef6ec;
  --navy-mid:     #fdf3e7;
  --navy-light:   #fceedd;
  --navy-card:    #fffaf4;

  --gold:         #b8860b;
  --gold-bright:  #9a6e00;
  --gold-dim:     #c9a84c;
  --gold-muted:   rgba(184,134,11,0.10);
  --gold-glow:    rgba(184,134,11,0.25);
  --gold-line:    rgba(184,134,11,0.20);
  --gold-border:  rgba(184,134,11,0.40);

  --crimson:      #c0392b;
  --crimson-dim:  rgba(192,57,43,0.10);
  --emerald:      #1e8449;
  --emerald-dim:  rgba(30,132,73,0.10);

  --parchment:    #6b4c11;
  --text:         #2c1a00;
  --text-2:       #7a5a1a;
  --text-3:       #b89040;

  --radius:       12px;
  --radius-sm:    7px;
  --shadow-royal: 0 4px 24px rgba(184,134,11,0.10), 0 1px 6px rgba(0,0,0,0.06);
  --shadow-gold:  0 4px 20px rgba(184,134,11,0.18);
}

/* ── Base reset ── */
html, body, .stApp {
  background: #fdf8f0 !important;
  font-family: 'EB Garamond', 'Cormorant Garamond', serif !important;
  color: var(--text) !important;
}

/* Royal parchment background */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% 0%, rgba(184,134,11,0.06) 0%, transparent 60%),
    repeating-linear-gradient(
      45deg,
      transparent,
      transparent 34px,
      rgba(184,134,11,0.025) 34px,
      rgba(184,134,11,0.025) 35px
    ),
    repeating-linear-gradient(
      -45deg,
      transparent,
      transparent 34px,
      rgba(184,134,11,0.025) 34px,
      rgba(184,134,11,0.025) 35px
    );
  pointer-events: none;
}

.stApp::after {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background:
    radial-gradient(ellipse 40% 60% at 0% 100%, rgba(184,134,11,0.05) 0%, transparent 50%),
    radial-gradient(ellipse 40% 60% at 100% 0%, rgba(184,134,11,0.05) 0%, transparent 50%);
  pointer-events: none;
}

.block-container {
  padding-top: 0 !important;
  max-width: 1380px !important;
  position: relative; z-index: 1;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Royal Cards ── */
.card {
  background: linear-gradient(145deg, #fffdf8 0%, #fff8ec 100%);
  border: 1px solid var(--gold-line);
  border-radius: var(--radius);
  padding: 1.8rem 2rem;
  margin: .7rem 0;
  box-shadow: var(--shadow-royal);
  position: relative;
  transition: border-color .3s, box-shadow .3s;
}
.card::before {
  content: '';
  position: absolute; top: 0; left: 50%; transform: translateX(-50%);
  width: 60%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}
.card:hover {
  border-color: var(--gold-border);
  box-shadow: var(--shadow-royal), 0 0 30px rgba(201,168,76,0.08);
}

/* ── Section Heading with royal crest style ── */
.sec-head {
  display: flex; align-items: center; gap: .9rem;
  font-family: 'Cinzel', serif;
  font-size: .68rem; font-weight: 600; color: var(--gold);
  letter-spacing: 3.5px; text-transform: uppercase;
  margin: 2rem 0 1rem;
}
.sec-head::before, .sec-head::after {
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold-line), transparent);
}
.sec-head::before { background: linear-gradient(90deg, transparent, var(--gold-dim)); }
.sec-head::after  { background: linear-gradient(90deg, var(--gold-dim), transparent); }

/* ── Input Fields ── */
.stTextInput input, .stNumberInput input,
.stTextArea textarea, .stDateInput input {
  background: #fffdf8 !important;
  color: var(--text) !important;
  border: 1px solid var(--gold-line) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'EB Garamond', serif !important;
  font-size: 1rem !important;
  transition: border-color .25s, box-shadow .25s !important;
}
.stTextInput input:focus, .stNumberInput input:focus,
.stTextArea textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(201,168,76,0.12) !important;
  outline: none !important;
}
div[data-baseweb="select"] > div {
  background: #fffdf8 !important;
  border: 1px solid var(--gold-line) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: 'EB Garamond', serif !important;
}
label {
  color: var(--text-2) !important;
  font-family: 'Cinzel', serif !important;
  font-size: .68rem !important;
  font-weight: 600 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}

/* ══════════════════════════════════════════
   ROYAL BUTTONS — Consistent Gold Standard
   ══════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, #8a6e2f 0%, #c9a84c 40%, #e8c96a 60%, #c9a84c 80%, #8a6e2f 100%) !important;
  color: #07090f !important;
  border: 1px solid rgba(232,201,106,0.5) !important;
  border-radius: var(--radius-sm) !important;
  padding: .75rem 1.6rem !important;
  font-family: 'Cinzel', serif !important;
  font-size: .78rem !important;
  font-weight: 700 !important;
  letter-spacing: 2.5px !important;
  text-transform: uppercase !important;
  width: 100% !important;
  transition: all .3s ease !important;
  box-shadow: 0 3px 16px rgba(201,168,76,0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important;
  position: relative !important;
  overflow: hidden !important;
  text-shadow: 0 1px 2px rgba(255,255,255,0.15) !important;
}
.stButton > button::before {
  content: '' !important;
  position: absolute !important; inset: 0 !important;
  background: linear-gradient(135deg, transparent 30%, rgba(255,255,255,0.12) 50%, transparent 70%) !important;
  transform: translateX(-100%) !important;
  transition: transform .5s ease !important;
}
.stButton > button:hover::before {
  transform: translateX(100%) !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #9a7e3f 0%, #d9b85c 40%, #f8d97a 60%, #d9b85c 80%, #9a7e3f 100%) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(201,168,76,0.45), inset 0 1px 0 rgba(255,255,255,0.25) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
  box-shadow: 0 2px 8px rgba(201,168,76,0.3) !important;
}

/* Download button — deeper gold variant */
.stDownloadButton > button {
  background: linear-gradient(135deg, #0a2e18 0%, #1a6a38 40%, #2a8a50 60%, #1a6a38 80%, #0a2e18 100%) !important;
  color: #c9e8c8 !important;
  border: 1px solid rgba(74,222,128,0.3) !important;
  border-radius: var(--radius-sm) !important;
  padding: .75rem 1.6rem !important;
  font-family: 'Cinzel', serif !important;
  font-size: .78rem !important;
  font-weight: 700 !important;
  letter-spacing: 2.5px !important;
  text-transform: uppercase !important;
  width: 100% !important;
  transition: all .3s ease !important;
  box-shadow: 0 3px 16px rgba(30,132,73,0.3), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}
.stDownloadButton > button:hover {
  background: linear-gradient(135deg, #0d3a1e 0%, #1f7a42 40%, #32a060 60%, #1f7a42 80%, #0d3a1e 100%) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(30,132,73,0.45) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
  background: #fffdf8 !important;
  border: 2px dashed var(--gold-line) !important;
  border-radius: var(--radius) !important;
  padding: 1.6rem !important;
  transition: border-color .3s, box-shadow .3s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--gold) !important;
  box-shadow: 0 0 24px var(--gold-muted) !important;
}

/* ── Diagnosis Badges ── */
.diag-badge {
  display: inline-flex; align-items: center; gap: .5rem;
  font-family: 'Cinzel', serif;
  font-size: 1.3rem; font-weight: 700;
  padding: .5rem 1.2rem; border-radius: 4px;
  letter-spacing: 1px;
}
.diag-normal    {
  background: var(--emerald-dim);
  color: #4ade80;
  border: 1px solid rgba(74,222,128,0.35);
  box-shadow: 0 0 20px rgba(74,222,128,0.1);
}
.diag-parkinson {
  background: var(--crimson-dim);
  color: #f87171;
  border: 1px solid rgba(248,113,113,0.35);
  box-shadow: 0 0 20px rgba(248,113,113,0.1);
}

/* ── Royal Stat Tiles ── */
.stat-tile {
  background: linear-gradient(145deg, #fffdf8, #fff6e8);
  border: 1px solid var(--gold-line);
  border-radius: var(--radius);
  padding: 1.1rem 1.3rem;
  text-align: center;
  box-shadow: var(--shadow-royal);
  position: relative;
}
.stat-tile::after {
  content: '';
  position: absolute; bottom: 0; left: 10%; right: 10%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}
.stat-value {
  font-family: 'Playfair Display', serif;
  font-size: 2rem; font-weight: 700; color: #9a6e00;
  line-height: 1.1; margin-bottom: .3rem;
}
.stat-label {
  font-family: 'Cinzel', serif;
  font-size: .58rem; color: var(--text-2);
  letter-spacing: 2px; text-transform: uppercase;
}

/* ── Risk Badges ── */
.risk-low      { background: rgba(74,222,128,.1);  color: #4ade80; border: 1px solid rgba(74,222,128,.3);  border-radius: 4px; padding: .25rem 1rem; font-size: .75rem; font-family: 'Cinzel',serif; font-weight: 600; letter-spacing: 1px; }
.risk-moderate { background: rgba(251,191,36,.1);  color: #fbbf24; border: 1px solid rgba(251,191,36,.3);  border-radius: 4px; padding: .25rem 1rem; font-size: .75rem; font-family: 'Cinzel',serif; font-weight: 600; letter-spacing: 1px; }
.risk-high     { background: rgba(248,113,113,.1); color: #f87171; border: 1px solid rgba(248,113,113,.3); border-radius: 4px; padding: .25rem 1rem; font-size: .75rem; font-family: 'Cinzel',serif; font-weight: 600; letter-spacing: 1px; }

/* ── Probability Bars ── */
.prob-row { margin: .6rem 0; }
.prob-label { font-family: 'Cinzel',serif; font-size: .65rem; color: var(--text-2); margin-bottom: .3rem; display: flex; justify-content: space-between; letter-spacing: 1px; }
.prob-track { background: rgba(255,255,255,.04); border-radius: 3px; height: 7px; overflow: hidden; border: 1px solid rgba(201,168,76,0.1); }
.prob-fill-n { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #065f46, #4ade80); transition: width .9s cubic-bezier(.22,1,.36,1); }
.prob-fill-p { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #7f1d1d, #f87171); transition: width .9s cubic-bezier(.22,1,.36,1); }

/* ── Image Frame ── */
.img-frame {
  border: 1px solid var(--gold-line);
  border-radius: var(--radius);
  overflow: hidden;
  background: #f5f0e8;
  box-shadow: var(--shadow-royal);
}
.img-caption {
  font-family: 'Cinzel', serif;
  font-size: .6rem; color: #b89040;
  text-align: center; padding: .45rem 0 .25rem;
  letter-spacing: 1.5px; text-transform: uppercase;
  background: rgba(245,240,232,0.8);
}

/* ── Heatmap Legend ── */
.hm-legend {
  display: flex; align-items: center; gap: .6rem;
  font-family: 'Cinzel', serif;
  font-size: .6rem; color: #b89040;
  margin-top: .6rem; letter-spacing: 1px;
}
.hm-bar { flex: 1; height: 6px; border-radius: 3px; background: linear-gradient(90deg,#00008b,#0000ff,#00ff00,#ffff00,#ff0000); }

/* ── Royal Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: #fffdf8 !important;
  border: 1px solid var(--gold-line) !important;
  border-radius: var(--radius) !important;
  padding: .35rem !important; gap: .25rem !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Cinzel', serif !important;
  font-size: .75rem !important; font-weight: 600 !important;
  color: #7a5a1a !important; border-radius: var(--radius-sm) !important;
  padding: .6rem 1.3rem !important; letter-spacing: 1.5px !important;
  transition: all .25s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #9a6e00 !important; }
.stTabs [aria-selected="true"] {
  background: rgba(184,134,11,0.10) !important;
  color: #7a4f00 !important;
  border: 1px solid rgba(184,134,11,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Royal Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #fdf8f0, #fdf3e7) !important;
  border-right: 1px solid var(--gold-line) !important;
}
[data-testid="stSidebar"] * { font-family: 'EB Garamond', serif !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'Cinzel', serif !important;
  color: #9a6e00 !important; letter-spacing: 2px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: linear-gradient(145deg, #fffdf8, #fff6e8) !important;
  border: 1px solid var(--gold-line) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.6rem !important; font-weight: 700 !important;
  color: #9a6e00 !important;
}
[data-testid="stMetricLabel"] {
  color: #7a5a1a !important;
  font-family: 'Cinzel', serif !important;
  font-size: .58rem !important; letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
}

/* ── Progress Bar ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--gold-dim), var(--gold-bright)) !important;
  border-radius: 3px !important;
}

/* ── DataTable ── */
[data-testid="stDataFrame"] {
  border-radius: var(--radius) !important;
  border: 1px solid var(--gold-line) !important;
  overflow: hidden !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #fdf8f0; }
::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 10px; }

/* ── HR / Dividers ── */
hr {
  border: none !important; height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--gold-line), transparent) !important;
  margin: 1.8rem 0 !important;
}

/* ── Alerts ── */
.stAlert { border-radius: var(--radius) !important; font-family: 'EB Garamond', serif !important; }

/* ── About Cards ── */
.about-card {
  background: linear-gradient(145deg, #fffdf8, #fff6e8);
  border: 1px solid var(--gold-line);
  border-radius: var(--radius);
  padding: 1.8rem 1.6rem;
  text-align: center;
  height: 100%;
  transition: border-color .3s, transform .3s;
  position: relative;
}
.about-card::before {
  content: '';
  position: absolute; top: 0; left: 15%; right: 15%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold-dim), transparent);
}
.about-card:hover {
  border-color: var(--gold-border);
  transform: translateY(-3px);
}

/* ── Team Card ── */
.team-card {
  background: linear-gradient(145deg, #fffdf8, #fff6e8);
  border: 1px solid var(--gold-line);
  border-radius: var(--radius);
  padding: 1.4rem 1rem;
  text-align: center;
  transition: border-color .3s, transform .3s;
}
.team-card:hover {
  border-color: var(--gold-border);
  transform: translateY(-4px);
  box-shadow: var(--shadow-gold);
}

/* ── Batch Result Rows ── */
.batch-normal    { border-left: 3px solid #4ade80; }
.batch-parkinson { border-left: 3px solid #f87171; }

/* ── Ornamental Divider ── */
.ornament {
  text-align: center;
  color: var(--gold-dim);
  font-size: 1.1rem;
  letter-spacing: 8px;
  margin: .5rem 0;
  font-family: 'Cinzel', serif;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ROYAL HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
logo_src  = get_logo_b64("logo.png")
logo_html = (
    f'<img src="{logo_src}" style="width:80px;height:80px;object-fit:contain;'
    f'border-radius:50%;border:2px solid rgba(184,134,11,0.5);'
    f'box-shadow:0 0 20px rgba(184,134,11,0.2);"/>'
    if logo_src else
    '<div style="width:80px;height:80px;background:linear-gradient(145deg,#fdf3dc,#fce8b8);'
    'border:2px solid rgba(184,134,11,0.5);border-radius:50%;display:flex;align-items:center;'
    'justify-content:center;font-size:2.2rem;box-shadow:0 0 20px rgba(184,134,11,0.2);">🧠</div>'
)

st.markdown(
    '<div style="'
    'background:linear-gradient(180deg,#fdf3dc 0%,#fdf8f0 60%,#fdf8f0 100%);'
    'border-bottom:1px solid rgba(184,134,11,0.22);'
    'padding:2.8rem 2rem 2.4rem;'
    'display:flex;flex-direction:column;align-items:center;gap:1rem;'
    'position:relative;overflow:hidden;">'

    # top golden rule
    '<div style="position:absolute;top:0;left:0;right:0;height:3px;'
    'background:linear-gradient(90deg,transparent,#c9a84c,#e8c96a,#c9a84c,transparent);"></div>'

    # corner ornaments
    '<div style="position:absolute;top:14px;left:24px;font-size:.9rem;'
    'color:rgba(184,134,11,0.4);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'
    '<div style="position:absolute;top:14px;right:24px;font-size:.9rem;'
    'color:rgba(184,134,11,0.4);font-family:serif;letter-spacing:4px;">✦ ✦ ✦</div>'

    + logo_html +

    '<div style="font-family:Cinzel,serif;font-size:2.8rem;font-weight:900;'
    'color:#2c1a00;letter-spacing:8px;line-height:1;text-align:center;'
    'text-shadow:0 2px 12px rgba(184,134,11,0.15);">'
    'NEUROSCAN&nbsp;<span style="color:#b8860b;">AI</span></div>'

    '<div style="font-family:Cormorant Garamond,serif;font-size:1rem;font-style:italic;'
    'color:#9a7030;letter-spacing:4px;text-align:center;">'
    'Parkinson\'s Detection · Brain MRI · Deep Learning</div>'

    # decorative line
    '<div style="display:flex;align-items:center;gap:1rem;width:60%;max-width:400px;">'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,#c9a84c);"></div>'
    '<span style="color:#b8860b;font-size:.7rem;letter-spacing:4px;">✦</span>'
    '<div style="flex:1;height:1px;background:linear-gradient(90deg,#c9a84c,transparent);"></div>'
    '</div>'

    # badges
    '<div style="display:flex;gap:1.2rem;flex-wrap:wrap;justify-content:center;">'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">⚙ ResNet50 + ViT</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">◈ 99.4% Accuracy</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">✦ Grad-CAM XAI</span>'
    '<span style="background:rgba(184,134,11,0.08);border:1px solid rgba(184,134,11,0.25);'
    'border-radius:3px;padding:.3rem 1rem;font-family:Cinzel,serif;font-size:.65rem;'
    'color:#8a5e00;letter-spacing:1.5px;">⬡ Batch Analysis</span>'
    '</div>'

    # bottom golden rule
    '<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
    'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'

    '</div>',
    unsafe_allow_html=True,
)

# ── Model download ─────────────────────────────────────────────────────────────
download_model()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("prediction_made",   False),
    ("patient_data",      {}),
    ("prediction_result", {}),
    ("batch_results",     []),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  ROYAL SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:.8rem 0 1.2rem;">'
        '<div style="font-family:Cinzel,serif;font-size:1.1rem;font-weight:700;'
        'color:#8a5e00;letter-spacing:3px;">⚕ NEUROSCAN AI</div>'
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.4),transparent);margin:.6rem 0;"></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Advanced deep-learning system for brain MRI analysis. "
        "Upload a scan to screen for Parkinson's Disease in seconds."
    )
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#9a7030;letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">⚙ Model Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.success(
        "**Architecture:** ResNet50 + ViT\n\n"
        "**Classes:** Normal / Parkinson's\n\n"
        "**Validation Accuracy:** ~99.4%\n\n"
        "**XAI:** Grad-CAM\n\n"
        "**Status:** 🟢 Online"
    )
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#9a7030;letter-spacing:2px;text-transform:uppercase;margin-bottom:.5rem;">✦ Capabilities</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:EB Garamond,serif;font-size:.95rem;color:#6b4c11;line-height:2;">'
        '⚜ Single-scan analysis<br>⚜ Batch processing<br>⚜ Grad-CAM heatmap<br>'
        '⚜ Risk scoring<br>⚜ PDF & CSV reports<br>⚜ Doctor field support'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.warning("⚠️ For **research/academic** purposes only. Not a substitute for clinical diagnosis.")
    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.25),transparent);margin:.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1: st.metric("Precision", "100%")
    with c2: st.metric("Recall",    "98%")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_scan, tab_batch, tab_about = st.tabs([
    "🧠  Single MRI Analysis",
    "📦  Batch Analysis",
    "🏛  About",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — SINGLE MRI
# ─────────────────────────────────────────────────────────────────────────────
with tab_scan:
    col_form, col_upload = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="sec-head">✦ Patient Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        patient_name = st.text_input("Full Name *", placeholder="Patient's full name")

        c1, c2 = st.columns(2)
        with c1: patient_age    = st.number_input("Age *", 0, 120, 45)
        with c2: patient_gender = st.selectbox("Gender *", ["Male", "Female", "Other"])

        c3, c4 = st.columns(2)
        with c3: patient_id  = st.text_input("Patient ID *", placeholder="P-2024-0001")
        with c4: scan_date   = st.date_input("Scan Date *", value=datetime.now())

        referring_doctor = st.text_input("Referring Doctor", placeholder="Dr. Name (optional)")
        medical_history  = st.text_area("Medical History", placeholder="Relevant history (optional)", height=90)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_upload:
        st.markdown('<div class="sec-head">✦ MRI Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["png", "jpg", "jpeg"],
            help="PNG / JPG / JPEG · any resolution",
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded MRI", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            w, h = image.size
            st.markdown(
                f'<div style="font-family:Cinzel,monospace;font-size:.62rem;'
                f'color:#9a7030;margin-top:.5rem;text-align:center;letter-spacing:1px;">'
                f'{image.mode} · {w}×{h}px · {uploaded_file.size/1024:.1f} KB</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("⬆️ Upload a brain MRI scan to begin analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2),transparent);margin:1.5rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-head">✦ Analysis & Results</div>', unsafe_allow_html=True)

    btn_col, report_col = st.columns([1, 1], gap="large")

    with btn_col:
        if st.button("⚕ Analyze MRI Scan"):
            if not patient_name.strip():
                st.error("⚠️ Please enter the patient's full name.")
            elif not patient_id.strip():
                st.error("⚠️ Please enter a Patient ID.")
            elif uploaded_file is None:
                st.error("⚠️ Please upload a brain MRI scan.")
            else:
                with st.spinner("🧠 Running AI analysis…"):
                    st.session_state.patient_data = {
                        "name":            patient_name.strip(),
                        "age":             patient_age,
                        "gender":          patient_gender,
                        "patient_id":      patient_id.strip(),
                        "scan_date":       scan_date.strftime("%Y-%m-%d"),
                        "doctor":          referring_doctor.strip() or "—",
                        "medical_history": medical_history.strip(),
                    }
                    try:
                        model, device = load_model()
                        result        = predict_single(model, device, image)
                        st.session_state.prediction_result = result
                        st.session_state.prediction_made   = True
                        st.success("✅ Analysis complete!")
                        st.balloons()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"❌ Error during analysis: {exc}")

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if st.session_state.prediction_made:
        r         = st.session_state.prediction_result
        is_normal = r["class_idx"] == 0
        diag_cls  = "diag-normal" if is_normal else "diag-parkinson"
        risk_cls  = f"risk-{r['risk_level'].lower()}"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Cinzel,serif;font-size:.72rem;font-weight:600;'
            'color:#8a5e00;letter-spacing:2.5px;text-transform:uppercase;'
            'text-align:center;margin-bottom:1.2rem;">◈ Diagnostic Summary ◈</div>',
            unsafe_allow_html=True,
        )

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Diagnosis</div>'
                f'<div style="margin-top:.5rem;"><span class="{diag_cls} diag-badge">'
                f'{"✦" if is_normal else "⚠"} {r["prediction"]}</span></div></div>',
                unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Confidence</div>'
                f'<div class="stat-value">{r["confidence"]:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Risk Level</div>'
                f'<div style="margin-top:.6rem;"><span class="{risk_cls}">{r["risk_level"]} Risk</span></div></div>',
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                f'<div class="stat-tile"><div class="stat-label">Timestamp</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.85rem;'
                f'color:#a89060;margin-top:.35rem;line-height:1.4;">{r["timestamp"]}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>✦ Normal</span><span>{r["normal_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-n" style="width:{r["normal_prob"]:.1f}%"></div></div>'
            f'</div>'
            f'<div class="prob-row">'
            f'<div class="prob-label"><span>⚠ Parkinson\'s</span><span>{r["parkinson_prob"]:.1f}%</span></div>'
            f'<div class="prob-track"><div class="prob-fill-p" style="width:{r["parkinson_prob"]:.1f}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Grad-CAM
        st.markdown('<div class="sec-head">✦ Grad-CAM Explainability</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:EB Garamond,serif;font-size:.95rem;color:#a89060;'
            'margin-bottom:1.2rem;line-height:1.7;">'
            'Grad-CAM highlights the brain regions that drove the model\'s decision. '
            '<strong style="color:#c9a84c;">Warmer colours (red/yellow)</strong> indicate higher neural attention.</p>',
            unsafe_allow_html=True,
        )
        ic1, ic2, ic3 = st.columns(3)
        for col, img_obj, cap in [
            (ic1, r["image"],       "Original MRI"),
            (ic2, r["cam_heatmap"], "Attention Heatmap"),
            (ic3, r["cam_overlay"], "Grad-CAM Overlay"),
        ]:
            with col:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(img_obj, use_column_width=True)
                st.markdown(f'<div class="img-caption">{cap}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="hm-legend"><span>Low</span><div class="hm-bar"></div><span>High</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with report_col:
        if st.session_state.prediction_made:
            if st.button("📜 Generate PDF Report"):
                with st.spinner("📝 Building royal report…"):
                    try:
                        pdf_bytes = build_pdf(
                            st.session_state.patient_data,
                            st.session_state.prediction_result,
                        )
                        p = st.session_state.patient_data
                        fname = f"NeuroScan_{p['patient_id']}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        st.download_button(
                            "⬇ Download PDF Report",
                            data=pdf_bytes,
                            file_name=fname,
                            mime="application/pdf",
                        )
                        st.success("✅ PDF ready for download!")
                    except Exception as exc:
                        st.error(f"❌ PDF generation error: {exc}")
        else:
            st.info("📋 Run an analysis first to generate a report.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — BATCH
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="sec-head">✦ Batch MRI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:EB Garamond,serif;font-size:1rem;color:#a89060;'
        'margin-bottom:1.2rem;line-height:1.7;">'
        'Upload multiple brain MRI scans at once. Each is independently analysed '
        'and results are summarised with a downloadable CSV report.</p>',
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "Upload Multiple Brain MRI Scans",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.info(f"📁 {len(batch_files)} file(s) ready for processing.")

        if st.button("⚕ Run Batch Analysis"):
            try:
                model, device = load_model()
            except Exception as exc:
                st.error(f"❌ Failed to load model: {exc}")
                st.stop()

            batch_results = []
            prog          = st.progress(0)
            status        = st.empty()

            for i, f in enumerate(batch_files):
                status.markdown(
                    f'<p style="font-family:Cinzel,serif;font-size:.75rem;color:#c9a84c;letter-spacing:1px;">'
                    f'Processing {f.name} ({i+1}/{len(batch_files)})…</p>',
                    unsafe_allow_html=True,
                )
                try:
                    res = predict_single(model, device, Image.open(f))
                    res["filename"] = f.name
                    batch_results.append(res)
                except Exception as exc:
                    st.warning(f"⚠️ Skipped {f.name}: {exc}")
                prog.progress((i + 1) / len(batch_files))

            st.session_state.batch_results = batch_results
            status.empty()
            st.success(f"✅ Done! {len(batch_results)} scan(s) processed.")
            st.rerun()

    if st.session_state.batch_results:
        results  = st.session_state.batch_results
        n_total  = len(results)
        n_normal = sum(1 for r in results if r["class_idx"] == 0)
        n_park   = n_total - n_normal
        avg_conf = np.mean([r["confidence"] for r in results])

        st.markdown('<div class="sec-head">✦ Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Total Scans",    n_total)
        with m2: st.metric("Normal",         n_normal)
        with m3: st.metric("Parkinson's",    n_park)
        with m4: st.metric("Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown('<div class="sec-head">✦ Distribution</div>', unsafe_allow_html=True)
        ch_col, tb_col = st.columns([1, 1], gap="large")

        with ch_col:
            fig, ax = plt.subplots(figsize=(4.5, 3.8), facecolor="#fdf8f0")
            ax.set_facecolor("#fdf8f0")
            if n_normal > 0 or n_park > 0:
                wedges, texts, autotexts = ax.pie(
                    [n_normal, n_park],
                    labels=["Normal", "Parkinson's"],
                    autopct="%1.0f%%",
                    colors=["#4ade80", "#f87171"],
                    startangle=90,
                    wedgeprops=dict(edgecolor="#0a0e1a", linewidth=2.5),
                    textprops=dict(color="#2c1a00", fontsize=10, fontfamily="EB Garamond"),
                )
                for at in autotexts:
                    at.set_color("#ffffff"); at.set_fontweight("bold")
            ax.set_title("Scan Distribution", color="#7a5a1a", fontsize=10, pad=10,
                         fontfamily="EB Garamond")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tb_col:
            df = pd.DataFrame([{
                "File":          r["filename"],
                "Prediction":    r["prediction"],
                "Confidence":    f"{r['confidence']:.1f}%",
                "Normal %":      f"{r['normal_prob']:.1f}%",
                "Parkinson's %": f"{r['parkinson_prob']:.1f}%",
                "Risk":          r["risk_level"],
            } for r in results])
            st.dataframe(df, use_container_width=True, height=280)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV Report",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        st.markdown('<div class="sec-head">✦ Per-Image Results</div>', unsafe_allow_html=True)
        for r in results:
            is_n = r["class_idx"] == 0
            col  = "#4ade80" if is_n else "#f87171"
            icon = "✦" if is_n else "⚠"
            rc1, rc2, rc3, rc4 = st.columns([1, 2, 1, 1])
            with rc1:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(r["image"], use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with rc2:
                st.markdown(
                    f'<p style="font-family:Cinzel,serif;font-size:.62rem;'
                    f'color:#9a7030;margin-bottom:.3rem;letter-spacing:1px;">{r["filename"]}</p>'
                    f'<p style="font-family:Playfair Display,serif;font-size:1.4rem;'
                    f'font-weight:700;color:{col};margin:0;">{icon} {r["prediction"]}</p>'
                    f'<p style="font-family:Cinzel,serif;font-size:.6rem;'
                    f'color:#9a7030;margin-top:.35rem;letter-spacing:1px;">{r["timestamp"]}</p>',
                    unsafe_allow_html=True,
                )
            with rc3:
                st.metric("Confidence", f"{r['confidence']:.1f}%")
                st.metric("Normal %",   f"{r['normal_prob']:.1f}%")
            with rc4:
                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                st.image(r["cam_overlay"], use_column_width=True)
                st.markdown('<div class="img-caption">Grad-CAM</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.15),transparent);margin:.8rem 0;"></div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — ABOUT  (Enhanced Royal Edition)
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    college_logo = get_logo_b64("bvcr.jpg")
    clg_html     = (
        f'<img src="{college_logo}" style="width:110px;height:110px;object-fit:contain;'
        f'margin-bottom:1.2rem;border:2px solid rgba(184,134,11,0.45);border-radius:10px;'
        f'box-shadow:0 6px 24px rgba(184,134,11,0.18);"/>'
        if college_logo else
        '<div style="font-size:4rem;margin-bottom:1.2rem;">🏛</div>'
    )

    # ── COLLEGE HERO BANNER ───────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(160deg,#fdf3dc 0%,#fdf8f0 50%,#fce8b0 100%);'
        f'border:1px solid rgba(184,134,11,0.28);border-radius:16px;'
        f'padding:3rem 2.5rem 2.5rem;margin-bottom:2rem;text-align:center;position:relative;overflow:hidden;">'

        # gold shimmer top bar
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,transparent,#c9a84c,#f0d070,#c9a84c,transparent);"></div>'

        # watermark pattern
        f'<div style="position:absolute;inset:0;opacity:0.03;font-size:8rem;'
        f'display:flex;align-items:center;justify-content:center;color:#b8860b;'
        f'font-family:Cinzel,serif;pointer-events:none;letter-spacing:20px;">✦BVC✦</div>'

        + clg_html +

        f'<div style="font-family:Cinzel,serif;font-size:1.9rem;font-weight:900;'
        f'color:#2c1a00;letter-spacing:4px;margin-bottom:.5rem;'
        f'text-shadow:0 2px 8px rgba(184,134,11,0.12);">BVC College of Engineering</div>'

        f'<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:1.05rem;'
        f'color:#b8860b;letter-spacing:3px;margin-bottom:.8rem;">Palacharla, Andhra Pradesh</div>'

        f'<div style="display:flex;align-items:center;gap:1rem;justify-content:center;margin-bottom:1rem;">'
        f'<div style="flex:1;max-width:120px;height:1px;background:linear-gradient(90deg,transparent,#c9a84c);"></div>'
        f'<span style="color:#b8860b;font-size:.8rem;letter-spacing:5px;">✦</span>'
        f'<div style="flex:1;max-width:120px;height:1px;background:linear-gradient(90deg,#c9a84c,transparent);"></div>'
        f'</div>'

        # accreditation badges
        f'<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.2rem;">'
        f'<span style="background:rgba(184,134,11,0.1);border:1px solid rgba(184,134,11,0.3);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#7a4f00;letter-spacing:1.5px;font-weight:600;">🎓 Autonomous</span>'
        f'<span style="background:rgba(30,132,73,0.08);border:1px solid rgba(30,132,73,0.25);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#1a6a38;letter-spacing:1.5px;font-weight:600;">✓ NAAC A Grade</span>'
        f'<span style="background:rgba(0,100,180,0.07);border:1px solid rgba(0,100,180,0.2);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#004a90;letter-spacing:1.5px;font-weight:600;">⚙ AICTE Approved</span>'
        f'<span style="background:rgba(140,40,40,0.07);border:1px solid rgba(140,40,40,0.2);'
        f'border-radius:4px;padding:.35rem 1.1rem;font-family:Cinzel,serif;font-size:.68rem;'
        f'color:#7a1a1a;letter-spacing:1.5px;font-weight:600;">📜 Affiliated to JNTUK</span>'
        f'</div>'

        f'<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a5a1a;'
        f'line-height:1.8;max-width:680px;margin:0 auto;">'
        f'BVC College of Engineering is a premier autonomous institution in Andhra Pradesh, '
        f'recognized for academic excellence and cutting-edge research. With state-of-the-art '
        f'laboratories, experienced faculty, and a legacy of producing distinguished engineers, '
        f'the institution stands as a beacon of quality technical education in the region.'
        f'</div>'

        f'<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
        f'background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── INSTITUTE STATS STRIP ─────────────────────────────────────────────────
    st.markdown('<div class="sec-head">✦ Institute at a Glance</div>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, num, label, icon in [
        (s1, "20+", "Years of Excellence", "🏆"),
        (s2, "3000+", "Students Enrolled", "👨‍🎓"),
        (s3, "150+", "Faculty Members", "👨‍🏫"),
        (s4, "10+", "Departments", "🏛"),
        (s5, "95%", "Placement Rate", "💼"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
                f'padding:1.2rem .8rem;text-align:center;'
                f'box-shadow:0 2px 12px rgba(184,134,11,0.08);">'
                f'<div style="font-size:1.6rem;margin-bottom:.3rem;">{icon}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.6rem;'
                f'font-weight:700;color:#9a6e00;line-height:1;">{num}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;color:#9a7030;'
                f'letter-spacing:1px;text-transform:uppercase;margin-top:.3rem;">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── PROJECT + TECH INFO CARDS ─────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2.2rem;">✦ About the Project</div>', unsafe_allow_html=True)

    pc1, pc2 = st.columns([3, 2], gap="large")
    with pc1:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
            'padding:2rem;height:100%;box-shadow:0 2px 16px rgba(184,134,11,0.08);">'

            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">⚗ Project Overview</div>'

            '<div style="font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;'
            'color:#2c1a00;margin-bottom:.9rem;line-height:1.3;">'
            'NeuroScan AI — Early Parkinson\'s Detection via Deep Learning</div>'

            '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#5a3a0a;'
            'line-height:1.85;margin-bottom:1.2rem;">'
            'NeuroScan AI is a B.Tech final-year capstone project developed at the Department of ECE, '
            'BVC College of Engineering. It applies a hybrid <strong style="color:#8a5e00;">ResNet50 + Vision Transformer (ViT)</strong> '
            'architecture to classify brain MRI scans and detect early signs of Parkinson\'s Disease with '
            'clinical-grade accuracy. The system integrates <strong style="color:#8a5e00;">Grad-CAM explainability</strong> '
            'to highlight regions of neural attention, bridging the gap between AI inference and medical '
            'interpretability.'
            '</div>'

            '<div style="display:flex;gap:.7rem;flex-wrap:wrap;">'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">🧠 Deep Learning</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">🔬 Medical Imaging</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">📊 Explainable AI</span>'
            '<span style="background:rgba(184,134,11,0.09);border:1px solid rgba(184,134,11,0.25);'
            'border-radius:4px;padding:.25rem .8rem;font-family:Cinzel,serif;font-size:.62rem;'
            'color:#7a4f00;letter-spacing:1px;">⚡ Computer Vision</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with pc2:
        st.markdown(
            '<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
            'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
            'padding:2rem;box-shadow:0 2px 16px rgba(184,134,11,0.08);">'

            '<div style="font-family:Cinzel,serif;font-size:.68rem;color:#b8860b;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:1rem;">⚙ Technology Stack</div>'

            # tech rows
            '<div style="display:flex;flex-direction:column;gap:.7rem;">'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">🤖</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">AI Framework</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'PyTorch · ResNet50 + ViT</div></div></div>'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">🔥</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Explainability</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'Gradient-weighted CAM</div></div></div>'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">🖥</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">User Interface</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'Streamlit · Custom CSS</div></div></div>'

            '<div style="display:flex;align-items:center;gap:.9rem;padding:.7rem;'
            'background:rgba(184,134,11,0.05);border-radius:8px;border-left:3px solid #c9a84c;">'
            '<span style="font-size:1.3rem;">📄</span>'
            '<div><div style="font-family:Cinzel,serif;font-size:.62rem;color:#8a5e00;'
            'letter-spacing:1px;font-weight:700;">Reporting</div>'
            '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#5a3a0a;">'
            'ReportLab · Pandas · CSV</div></div></div>'

            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── MODEL PERFORMANCE METRICS ─────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2rem;">✦ Model Performance</div>', unsafe_allow_html=True)
    pm1, pm2, pm3, pm4 = st.columns(4)
    for col, val, lbl, sub, color in [
        (pm1, "99.4%", "Validation Accuracy", "On held-out test set", "#7a4f00"),
        (pm2, "100%",  "Precision",           "No false positives",   "#1a6a38"),
        (pm3, "98%",   "Recall",              "High sensitivity",     "#004a90"),
        (pm4, "99.1%", "F1 Score",            "Harmonic mean",        "#7a1a1a"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:12px;'
                f'padding:1.4rem 1rem;text-align:center;'
                f'box-shadow:0 2px 12px rgba(184,134,11,0.07);position:relative;">'
                f'<div style="position:absolute;top:0;left:20%;right:20%;height:2px;'
                f'background:linear-gradient(90deg,transparent,{color},transparent);opacity:.4;"></div>'
                f'<div style="font-family:Playfair Display,serif;font-size:2rem;'
                f'font-weight:900;color:{color};line-height:1;">{val}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.65rem;color:#7a5a1a;'
                f'letter-spacing:1.5px;text-transform:uppercase;margin:.4rem 0 .2rem;">{lbl}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.82rem;color:#b89040;">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── TEAM SECTION ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Project Team</div>', unsafe_allow_html=True)

    team = [
        {"roll": "236M5A0408", "name": "G Srinivasu",      "icon": "👨‍💻", "role": "AI Model & Backend"},
        {"roll": "226M1A0460", "name": "S Anusha Devi",    "icon": "👩‍💻", "role": "UI/UX & Frontend"},
        {"roll": "226M1A0473", "name": "V V Siva Vardhan", "icon": "👨‍💻", "role": "Data & Testing"},
        {"roll": "236M5A0415", "name": "N L Sandeep",      "icon": "👨‍💻", "role": "Reports & Integration"},
    ]
    tcols = st.columns(4)
    for i, m in enumerate(team):
        with tcols[i]:
            st.markdown(
                f'<div class="team-card" style="position:relative;overflow:hidden;">'
                f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
                f'background:linear-gradient(90deg,transparent,#c9a84c,transparent);"></div>'
                f'<div style="font-size:2.6rem;margin-bottom:.6rem;">{m["icon"]}</div>'
                f'<div style="font-family:Playfair Display,serif;font-size:1.05rem;font-weight:700;'
                f'color:#2c1a00;margin-bottom:.3rem;">{m["name"]}</div>'
                f'<div style="font-family:EB Garamond,serif;font-size:.88rem;color:#8a5e00;'
                f'margin-bottom:.4rem;font-style:italic;">{m["role"]}</div>'
                f'<div style="font-family:Cinzel,serif;font-size:.58rem;'
                f'color:#b89040;letter-spacing:1.5px;">{m["roll"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── GUIDANCE SECTION ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:2.4rem;">✦ Project Guidance</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3, gap="medium")
    for col, role_label, role_color, icon, name, tags in [
        (g1, "Project Guide", "#b8860b", "👨‍🏫",
         "Ms. N P U V S N Pavan Kumar, M.Tech",
         [("Assistant Professor", "#8a5e00"), ("Dept. of ECE", "#8a5e00"), ("Deputy CoE – III", "#8a5e00")]),
        (g2, "Project Coordinator", "#1a6a38", "📋",
         "Mr. K Anji Babu, M.Tech",
         [("Assistant Professor", "#1a6a38"), ("Dept. of ECE", "#1a6a38")]),
        (g3, "Head of Department", "#7a1a1a", "👨‍💼",
         "Dr. S A Vara Prasad, Ph.D, M.Tech",
         [("Professor & HOD · ECE", "#7a1a1a"), ("Chairman BoS", "#7a1a1a"), ("Anti-Ragging Committee", "#7a1a1a")]),
    ]:
        tag_html = "".join(
            f'<span style="background:rgba({",".join(str(int(role_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},.08);'
            f'border:1px solid rgba({",".join(str(int(role_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},.25);'
            f'border-radius:4px;padding:.2rem .75rem;font-family:Cinzel,serif;font-size:.6rem;'
            f'color:{rc};letter-spacing:1px;font-weight:600;">{t}</span>'
            for t, rc in tags
        )
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#fffdf8,#fff6e8);'
                f'border:1px solid rgba(184,134,11,0.2);border-radius:14px;'
                f'padding:2rem 1.6rem;text-align:center;position:relative;'
                f'box-shadow:0 3px 18px rgba(184,134,11,0.08);height:100%;">'

                f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
                f'background:linear-gradient(90deg,transparent,{role_color},transparent);'
                f'border-radius:14px 14px 0 0;"></div>'

                f'<div style="font-family:Cinzel,serif;font-size:.6rem;color:{role_color};'
                f'letter-spacing:2.5px;text-transform:uppercase;margin-bottom:.9rem;font-weight:700;">'
                f'⭐ {role_label}</div>'

                f'<div style="font-size:2.5rem;margin-bottom:.7rem;">{icon}</div>'

                f'<div style="font-family:Playfair Display,serif;font-size:1rem;font-weight:700;'
                f'color:#2c1a00;margin-bottom:1rem;line-height:1.4;">{name}</div>'

                f'<div style="display:flex;flex-wrap:wrap;gap:.4rem;justify-content:center;">'
                f'{tag_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── DISCLAIMER ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(192,57,43,0.05),rgba(192,57,43,0.02));'
        'border:1px solid rgba(192,57,43,0.2);border-left:4px solid #c0392b;'
        'border-radius:10px;padding:1.2rem 1.8rem;">'
        '<div style="font-family:Cinzel,serif;font-size:.65rem;color:#c0392b;'
        'letter-spacing:2px;text-transform:uppercase;margin-bottom:.4rem;">⚕ Medical Disclaimer</div>'
        '<div style="font-family:EB Garamond,serif;font-size:1rem;color:#7a1a1a;line-height:1.7;">'
        'This project is developed <strong>for academic and research purposes only</strong>. '
        'The AI system is not a certified medical device and its outputs should never be used '
        'as a substitute for professional clinical diagnosis. '
        'Always consult a qualified neurologist for any medical decisions.'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ── ROYAL FOOTER ──────────────────────────────────────────────────────────────
st.markdown(
    '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(184,134,11,0.3),transparent);margin:2rem 0 0;"></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align:center;padding:1.6rem;background:linear-gradient(180deg,#fdf8f0,#fdf3dc);">'
    '<div style="font-family:Cinzel,serif;font-size:.6rem;color:#9a7030;'
    'letter-spacing:3px;text-transform:uppercase;margin-bottom:.5rem;">'
    'Research &amp; Educational Use Only · Not for Clinical Diagnosis'
    '</div>'
    '<div style="display:flex;align-items:center;gap:1rem;justify-content:center;margin-bottom:.5rem;">'
    '<div style="flex:1;max-width:80px;height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,0.2));"></div>'
    '<span style="color:rgba(201,168,76,0.3);font-size:.6rem;">✦</span>'
    '<div style="flex:1;max-width:80px;height:1px;background:linear-gradient(90deg,rgba(201,168,76,0.2),transparent);"></div>'
    '</div>'
    '<div style="font-family:EB Garamond,serif;font-size:.9rem;color:#9a7030;">'
    'NeuroScan AI · ResNet50 + ViT · Grad-CAM · PyTorch · Streamlit'
    '</div></div>',
    unsafe_allow_html=True,
)
