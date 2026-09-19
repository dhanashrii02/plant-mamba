"""
Plant Mamba - Executive Phytosanitary Diagnostic PDF Report Generator
Redesigned with FPDF2 for an un-congested, highly professional, 2-page clinical laboratory layout.
Page 1: Diagnostic Assessment, Quality Screening, High-Res Dual Imaging, Microclimate Risk
Page 2: Structured Agronomic Protocols (Chemical, Biological, Cultural), Regional Outbreak History & Certification
"""

import os
import tempfile
from datetime import datetime
from fpdf import FPDF
from fpdf.enums import XPos, YPos


class PlantMambaClinicalPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_id = "PMR-DEMO"

    def header(self):
        # Top Clinical Header Band
        self.set_fill_color(15, 60, 35)  # Deep forest emerald
        self.rect(0, 0, 210, 22, "F")
        
        # Gold accent line
        self.set_fill_color(212, 175, 55)
        self.rect(0, 22, 210, 1.5, "F")

        # Clinic Header Typography
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(12, 4)
        self.cell(186, 7, "PLANT MAMBA: PHYTOSANITARY CLINICAL DIAGNOSTIC REPORT", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        self.set_font("Helvetica", "", 8)
        self.set_text_color(200, 235, 210)
        self.cell(186, 4, "Autonomous Agricultural Pathology, Visual State Space Core & Explainable XAI",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(26)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(210, 220, 210)
        self.line(12, self.get_y(), 198, self.get_y())
        
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(120, 130, 120)
        self.set_y(-11)
        self.cell(90, 5, f"Report ID: {self.report_id} | Plant Mamba Clinical Pathology Engine", align="L")
        self.cell(96, 5, f"Page {self.page_no()} of {{nb}} | Official Confidential Document", align="R")


def generate_pdf_report(diagnosis_result, weather_info, risk_info, similar_cases, custom_report_id=None):
    """
    Generates an elegant, spacious, multi-page clinical laboratory diagnostic report.
    Returns the raw PDF bytes.
    """
    pdf = PlantMambaClinicalPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=False)  # Strict manual page formatting for clean layout

    now = datetime.now()
    report_id = custom_report_id or f"PMR-{now.strftime('%y%m%d%H%M%S')}"
    pdf.report_id = report_id

    city = weather_info.get("city", "Nagpur")
    temp = weather_info.get("temperature", 26.5)
    humidity = weather_info.get("humidity", 78.0)
    risk_level = risk_info.get("risk_level", "Moderate Risk")
    risk_explanation = risk_info.get("explanation", "Ambient humidity warrants routine field monitoring.")

    pred_disease = diagnosis_result.get("clean_name", diagnosis_result.get("predicted_class", "Unknown Pathology"))
    confidence = float(diagnosis_result.get("confidence", 0.0))
    severity_level = diagnosis_result.get("severity_level", "Moderate")
    infected_ratio = float(diagnosis_result.get("infected_ratio", 0.0))
    quality = diagnosis_result.get("quality", {})
    treatment = diagnosis_result.get("treatment", {})

    # =========================================================================
    # PAGE 1: DIAGNOSIS, HIGH-RES VISUALS & MICROCLIMATE
    # =========================================================================
    pdf.add_page()

    # 1. Document Metadata Card
    pdf.set_fill_color(248, 250, 248)
    pdf.set_draw_color(205, 220, 205)
    pdf.rect(12, 28, 186, 17, "FD")

    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(50, 60, 50)
    pdf.set_xy(16, 30)
    pdf.cell(46, 5, f"REPORT ID: {report_id}")
    pdf.cell(48, 5, f"DATE: {now.strftime('%d-%b-%Y %H:%M')}")
    pdf.cell(46, 5, f"LOCATION: {city}, India")
    pdf.cell(46, 5, f"MICROCLIMATE: {temp} C / {humidity}% RH")

    pdf.set_xy(16, 36)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 95, 80)
    pdf.cell(94, 5, f"SPECIMEN QUALITY: {quality.get('status', 'Passed')} (Sharpness: {quality.get('sharpness', 0):.1f} / Brightness: {quality.get('brightness', 0):.1f})")
    pdf.cell(92, 5, "DIAGNOSTIC CORE: Vision Mamba SSM + ResNet-18 (224px High-Res)")

    # 2. Section 1: Primary Clinical Diagnosis Box
    pdf.set_y(49)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 60, 35)
    pdf.cell(186, 6, "1. PRIMARY PATHOLOGICAL DIAGNOSIS & SEVERITY CLASSIFICATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Diagnostic Summary Box
    pdf.set_fill_color(242, 247, 244)
    pdf.set_draw_color(175, 205, 180)
    pdf.rect(12, 56, 186, 24, "FD")

    # Disease Title
    pdf.set_xy(16, 58)
    pdf.set_font("Helvetica", "B", 14)
    if "healthy" in pred_disease.lower():
        pdf.set_text_color(16, 100, 40)
    else:
        pdf.set_text_color(160, 25, 25)
    pdf.cell(178, 7, pred_disease, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Metrics row
    pdf.set_xy(16, 67)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(40, 50, 40)
    pdf.cell(45, 5, f"AI Confidence: {confidence:.2f}%")
    pdf.cell(50, 5, f"Pathology Stage: {severity_level}")
    pdf.cell(45, 5, f"Infected Leaf Tissue: {infected_ratio:.1f}%")
    
    status_tag = "HEALTHY SPECIMEN" if "healthy" in pred_disease.lower() else "TREATMENT REQUIRED"
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 245, 220) if "healthy" in pred_disease.lower() else pdf.set_fill_color(255, 230, 230)
    pdf.set_text_color(20, 100, 30) if "healthy" in pred_disease.lower() else pdf.set_text_color(180, 20, 20)
    pdf.cell(38, 5, f"  {status_tag}  ", border=1, fill=True, align="C")

    # 3. Section 2: Visual Explainable AI (Side-by-Side High-Res Images)
    pdf.set_y(85)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 60, 35)
    pdf.cell(186, 6, "2. VISUAL EXPLAINABLE AI & LESION-GUIDED SALIENCY LOCALIZATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Save images to temp directory for insertion
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_file = os.path.join(tmp_dir, "orig.jpg")
        cam_file = os.path.join(tmp_dir, "cam.jpg")

        orig_img = diagnosis_result.get("original_image")
        cam_img = diagnosis_result.get("cam_image")

        if orig_img is not None and cam_img is not None:
            orig_img.save(orig_file, "JPEG", quality=95)
            cam_img.save(cam_file, "JPEG", quality=95)

            img_size = 72  # 72mm x 72mm generous square display
            y_img = 93
            
            # Draw shaded image card backings
            pdf.set_fill_color(245, 245, 245)
            pdf.set_draw_color(210, 215, 210)
            pdf.rect(20, y_img - 1, img_size + 2, img_size + 2, "FD")
            pdf.rect(114, y_img - 1, img_size + 2, img_size + 2, "FD")

            pdf.image(orig_file, x=21, y=y_img, w=img_size, h=img_size)
            pdf.image(cam_file, x=115, y=y_img, w=img_size, h=img_size)

            # Captions
            pdf.set_y(y_img + img_size + 3)
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(50, 60, 50)
            pdf.cell(93, 4, "Figure 1: High-Resolution Leaf Specimen", align="C")
            pdf.cell(93, 4, "Figure 2: Lesion-Guided Grad-CAM Pathology Heatmap", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(110, 120, 110)
            pdf.cell(93, 4, "Natural RGB sensor capture", align="C")
            pdf.cell(93, 4, "Red/Yellow hotspots pinpoint physical necrotic lesions", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 4. Section 3: Microclimate Context & Environmental Risk
    pdf.set_y(183)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 60, 35)
    pdf.cell(186, 6, "3. LOCAL AGRO-METEOROLOGICAL EPIDEMIC RISK ANALYSIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Weather Risk Box
    if "Critical" in risk_level or "High" in risk_level:
        pdf.set_fill_color(255, 242, 242)
        pdf.set_draw_color(230, 160, 160)
        title_color = (180, 25, 25)
    else:
        pdf.set_fill_color(240, 249, 255)
        pdf.set_draw_color(175, 210, 235)
        title_color = (20, 70, 130)

    pdf.rect(12, 191, 186, 22, "FD")
    pdf.set_xy(16, 193)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(*title_color)
    pdf.cell(178, 5, f"Atmospheric Infection Risk Index: {risk_level.upper()} ({temp} C | {humidity}% RH)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(16, 199)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 45, 40)
    pdf.multi_cell(178, 4.5, f"Field Impact: {risk_explanation} High relative humidity combined with temperature between 18C and 28C accelerates fungal spore germination on leaf tissue.")

    # Page 1 Footer Transition Note
    pdf.set_y(220)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 110, 100)
    pdf.cell(186, 5, "Please proceed to Page 2 for detailed chemical, biological, and cultural treatment prescriptions.", align="C")

    # =========================================================================
    # PAGE 2: DETAILED PRESCRIPTIONS, REGIONAL MATCHING & CERTIFICATION
    # =========================================================================
    pdf.add_page()

    # Section 4: Comprehensive Agronomic Remediation Protocols
    pdf.set_y(28)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 60, 35)
    pdf.cell(186, 6, "4. CLINICAL AGRONOMIC PRESCRIPTIONS & MANAGEMENT PROTOCOLS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 4.1 Chemical Remediation Protocol Card
    pdf.set_y(36)
    pdf.set_fill_color(255, 247, 247)
    pdf.set_draw_color(225, 180, 180)
    pdf.rect(12, 36, 186, 38, "FD")

    pdf.set_xy(16, 38)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(160, 25, 25)
    pdf.cell(178, 5, "[A] Chemical Remediation & Fungicidal / Bactericidal Formulation:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    chem_text = treatment.get("treatment", "Apply recommended protective fungicides such as Copper Oxychloride or Mancozeb as per standard regional agronomic dosage.")
    pdf.set_xy(16, 44)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(178, 4.5, chem_text)

    pdf.set_xy(16, 64)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(120, 60, 60)
    pdf.cell(178, 4, "* Note: Ensure protective gear during spraying. Comply with standard 14-day pre-harvest intervals.")

    # 4.2 Biological & Organic Controls Card
    pdf.set_y(78)
    pdf.set_fill_color(245, 252, 245)
    pdf.set_draw_color(180, 220, 180)
    pdf.rect(12, 78, 186, 38, "FD")

    pdf.set_xy(16, 80)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(20, 100, 35)
    pdf.cell(178, 5, "[B] Biological & Organic Remediation Protocol:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_xy(16, 86)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 30, 30)
    bio_text = ("Apply bio-fungicides such as Trichoderma viride or Pseudomonas fluorescens @ 5g/L of water. "
                "Foliar spray of 5% cold-pressed neem seed kernel extract (NSKE) disrupts fungal sporulation without harming beneficial pollinators.")
    pdf.multi_cell(178, 4.5, bio_text)

    pdf.set_xy(16, 106)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(60, 110, 60)
    pdf.cell(178, 4, "* Certified organic and residue-free practice for sustainable agricultural soil biology.")

    # 4.3 Cultural & Preventive Management Card
    pdf.set_y(120)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(190, 205, 220)
    pdf.rect(12, 120, 186, 36, "FD")

    pdf.set_xy(16, 122)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.set_text_color(30, 70, 120)
    pdf.cell(178, 5, "[C] Cultural Sanitation & Preventive Agronomy:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    prev_text = treatment.get("prevention", "Maintain adequate crop row spacing to maximize canopy airflow. Transition from overhead sprinklers to drip irrigation to keep foliar tissue dry.")
    pdf.set_xy(16, 128)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(178, 4.5, prev_text)

    # Section 5: Historical Regional Outbreak Correlation
    pdf.set_y(162)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 60, 35)
    pdf.cell(186, 6, "5. REGIONAL EPIDEMIOLOGICAL OUTBREAK CORRELATION (MAHARASHTRA DATABASE)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(252, 252, 252)
    pdf.set_draw_color(215, 220, 215)
    pdf.rect(12, 169, 186, 26, "FD")

    pdf.set_xy(16, 171)
    pdf.set_font("Helvetica", "", 8.2)
    pdf.set_text_color(40, 50, 40)
    if similar_cases:
        for idx, case in enumerate(similar_cases[:2]):
            pdf.cell(178, 4.5, f"* Case Ref: {case['case_id']} | Pathogen: {case['disease']} | Match Similarity: {case['match_pct']}% | Region: {case['region']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_x(20)
            pdf.set_font("Helvetica", "I", 7.8)
            pdf.cell(174, 4, f"Observed Field Outcome: {case['outcome']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 8.2)
            pdf.set_x(16)
    else:
        pdf.cell(178, 5, "* No high-similarity regional outbreaks registered in the database for this specific crop season.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Clean, simple professional footer note
    pdf.set_y(205)
    pdf.set_draw_color(210, 220, 210)
    pdf.line(12, 205, 198, 205)
    pdf.set_y(208)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(110, 120, 110)
    pdf.cell(186, 5, "Plant Mamba Phytosanitary Diagnostic Report - Certified for Field Decision Support", align="C")

    return bytes(pdf.output())


if __name__ == "__main__":
    print("Testing PlantMambaClinicalPDF compilation...")
    from PIL import Image
    test_img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    diag = {
        "clean_name": "Apple Cedar Rust",
        "predicted_class": "Apple___Cedar_apple_rust",
        "confidence": 96.4,
        "severity_level": "Moderate",
        "infected_ratio": 13.1,
        "quality": {"status": "Passed", "sharpness": 88.5, "brightness": 120.0},
        "original_image": test_img,
        "cam_image": test_img,
        "treatment": {
            "treatment": "Apply myclobutanil or mancozeb at bud break. Repeat every 10-14 days until petal fall.",
            "prevention": "Remove nearby cedar galls within 1-2 miles. Select rust-resistant apple cultivars."
        }
    }
    weather = {"city": "Nagpur", "temperature": 27.2, "humidity": 82.0}
    risk = {"risk_level": "High Risk", "explanation": "High humidity > 80% accelerates fungal rust sporulation."}
    cases = [{"case_id": "CS-NAG-2401", "disease": "Apple Cedar Rust", "match_pct": 94.5, "region": "Nagpur (East)", "outcome": "Full control achieved via myclobutanil."}]
    pdf_bytes = generate_pdf_report(diag, weather, risk, cases)
    print(f"Generated PDF successfully! Size: {len(pdf_bytes)} bytes.")
