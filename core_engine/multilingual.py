"""
Multilingual Localization System
Supports English, Hindi (हिन्दी), and Marathi (मराठी) across all 38 plant disease classes.
"""

from vision_mamba_model import CLEAN_CLASS_NAMES

TRANSLATIONS = {
    "en": {
        "title": "🌿 Plant Mamba: Visual Intelligence for Crop Health",
        "sub_title": "Powered by ResNet-18 + Bidirectional SSM (Vision Mamba), Grad-CAM Explainability & Agronomic Protocols",
        "quality_passed": "Image resolution, focus, and illumination meet clinical requirements.",
        "quality_warning": "Suboptimal image quality detected. Sharpness or illumination may affect precision.",
        "predicted_disease": "Predicted Disease",
        "confidence_score": "Confidence Score",
        "severity_level": "Severity Level",
        "infected_leaf_area": "Infected Leaf Area",
        "weather_context": "⛅ Weather Context & Risk",
        "similar_cases": "🔍 Similar Historical Cases Retrieved",
        "treatment_recommendations": "💊 Treatment & Management Recommendations",
        "chem_bio_treatment": "Chemical / Bio Treatment",
        "cultural_management": "Cultural / Organic Management",
        "clinical_note": "Clinical Impact Note",
        "xai_header": "🔍 Visual Explainable AI (Grad-CAM) & Lesion Localization",
        "original_photo": "Original Leaf Photo",
        "attention_heatmap": "Attention Heatmap Overlay",
        "heatmap_caption": "Warm regions (red/yellow) indicate key lesion zones driving the neural diagnosis.",
        "top_alternatives": "📊 View Top 3 Differential Class Probabilities",
        "download_pdf_btn": "📄 Download Clinical Diagnostic Report (PDF)",
        "generating_pdf": "Generating Clinical PDF Report...",
        "language_select": "Select Language",
        "region_select": "Select Farm Region",
        "upload_label": "Upload Leaf Image (JPG/PNG)",
        "sample_select_label": "Test with Pre-loaded Sample Leaf:",
        "risk_level_label": "Risk Level",
        "match_label": "Match",
        "region_label": "Region",
        "nav_heading": "Navigation",
        "new_diagnosis": "➕ New Diagnosis",
        "farm_location_weather": "Farm Location & Weather",
        "recent_consultations": "Recent Consultations",
        "all_field_submissions": "All Field Submissions",
        "clear_history": "Clear All History",
        "no_consultations": "No consultations found. Run your first leaf scan above!",
        "logout": "🚪 Logout",
        "logged_in_as": "Logged in as:",
        "chief_agronomist_tag": "🩺 Chief Agronomist • Clinical Review Authority",
        "farmer_tag": "🚜 Progressive Farmer • Field Production & Spray Assistant",
        "nav_scanner": "🌿 AI Diagnostic Scanner",
        "nav_review": "🩺 Clinical Review & Sign-Off",
        "nav_spray_calendar": "🚜 Spray Calendar & Tank-Mix Calculator",
        "confidence_label": "Diagnostic Confidence",
        "severity_label": "Pathology Severity",
        "infected_area_label": "Infected Leaf Area",
        "microclimate_risk": "Microclimate Risk",
        "treatment_heading": "Chemical & Biological Treatment",
        "prevention_heading": "Cultural Sanitation & Prevention",
        "historical_heading": "Historical Epidemiological Cases",
        "submitted_leaf": "🍃 Submitted Leaf Specimen",
        "gradcam_heatmap": "🔥 Lesion-Guided Grad-CAM Heatmap",
        "download_pdf": "📄 Download Official Phytosanitary PDF Report",
        "diagnosis_confirmed": "CLINICAL DIAGNOSIS CONFIRMED",
        "select_sample_prompt": "👈 Please select a sample leaf image or upload your own leaf photo to begin diagnosis.",
        "supported_crops_heading": "Supported Crops & Pathologies (38 Categories):",
        "none_upload_own": "None (Upload your own)",
        "sample_prefix": "Sample",
        "agro_open_review_btn": "🩺 Open Clinical Review & Audit Suite ➔",
        "farmer_open_spray_btn": "🚜 Calculate Tank-Mix Dosage & 14-Day Spray Calendar ➔",
        "report_id_label": "Report ID",
        "location_label": "Location",
        "model_core_label": "Model: Vision Mamba SSM (224px)",
    },
    "hi": {
        "title": "🌿 प्लांट मांबा: फसल स्वास्थ्य के लिए विजुअल इंटेलिजेंस",
        "sub_title": "विज़न मांबा (ResNet-18 + बायो-डायरेक्शनल SSM), ग्रैड-कैम एवं कृषि विज्ञान प्रोटोकॉल द्वारा संचालित",
        "quality_passed": "छवि का रिज़ॉल्यूशन, फोकस और रोशनी नैदानिक आवश्यकताओं के पूर्णतः अनुकूल है।",
        "quality_warning": "छवि गुणवत्ता में मामूली कमी। धुंधलापन या प्रकाश विश्लेषण की सटीकता को प्रभावित कर सकता है।",
        "predicted_disease": "पहचाना गया रोग",
        "confidence_score": "सटीकता स्कोर",
        "severity_level": "गंभीरता स्तर",
        "infected_leaf_area": "संक्रमित पत्ती क्षेत्र",
        "weather_context": "⛅ स्थानीय मौसम एवं रोग जोखिम",
        "similar_cases": "🔍 पूर्व ऐतिहासिक नैदानिक मामले",
        "treatment_recommendations": "💊 उपचार एवं प्रबंधन संबंधी सिफारिशें",
        "chem_bio_treatment": "रासायनिक / जैविक उपचार",
        "cultural_management": "जैविक एवं निवारक प्रबंधन",
        "clinical_note": "रोग प्रभाव संबंधी नोट",
        "xai_header": "🔍 विजुअल एक्सप्लेनेबल एआई (Grad-CAM) एवं रोगग्रस्त क्षेत्र",
        "original_photo": "मूल पत्ती की तस्वीर",
        "attention_heatmap": "ग्रैड-कैम हीटमैप ओवरले",
        "heatmap_caption": "लाल और पीले क्षेत्र उन प्रमुख संक्रमित भागों को दर्शाते हैं जिन पर मॉडल ने निर्णय लिया।",
        "top_alternatives": "📊 शीर्ष 3 संभावित रोगों का वितरण",
        "download_pdf_btn": "📄 नैदानिक रिपोर्ट डाउनलोड करें (PDF)",
        "generating_pdf": "नैदानिक पीडीएफ रिपोर्ट तैयार हो रही है...",
        "language_select": "भाषा चुनें",
        "region_select": "अपना कृषि क्षेत्र चुनें",
        "upload_label": "पत्ती की तस्वीर अपलोड करें (JPG/PNG)",
        "sample_select_label": "परीक्षण के लिए नमूना पत्ती चुनें:",
        "risk_level_label": "जोखिम स्तर",
        "match_label": "समानता",
        "region_label": "क्षेत्र",
        "nav_heading": "नेविगेशन",
        "new_diagnosis": "➕ नया निदान",
        "farm_location_weather": "खेत का स्थान एवं मौसम",
        "recent_consultations": "हाल के नैदानिक रिकॉर्ड",
        "all_field_submissions": "सभी क्षेत्रीय फसल रिकॉर्ड",
        "clear_history": "सारा इतिहास हटाएं",
        "no_consultations": "कोई पिछला रिकॉर्ड नहीं मिला। ऊपर अपनी पहली पत्ती स्कैन करें!",
        "logout": "🚪 लॉग आउट",
        "logged_in_as": "लॉग इन:",
        "chief_agronomist_tag": "🩺 मुख्य कृषि विशेषज्ञ • नैदानिक समीक्षा प्राधिकरण",
        "farmer_tag": "🚜 प्रगतिशील किसान • फसल सुरक्षा एवं छिड़काव सहायक",
        "nav_scanner": "🌿 एआई फसल रोग स्कैनर",
        "nav_review": "🩺 नैदानिक समीक्षा एवं प्रमाणन",
        "nav_spray_calendar": "🚜 छिड़काव कैलेंडर और खुराक कैलकुलेटर",
        "confidence_label": "नैदानिक सटीकता",
        "severity_label": "संक्रमण गंभीरता",
        "infected_area_label": "संक्रमित पत्ती क्षेत्र",
        "microclimate_risk": "मौसम आधारित जोखिम",
        "treatment_heading": "रासायनिक एवं जैविक उपचार",
        "prevention_heading": "कृषि स्वच्छता एवं रोकथाम उपाय",
        "historical_heading": "क्षेत्रीय ऐतिहासिक प्रकोप मामले",
        "submitted_leaf": "🍃 जांची गई पत्ती का नमूना",
        "gradcam_heatmap": "🔥 ग्रैड-कैम रोग संक्रमण हीटमैप",
        "download_pdf": "📄 आधिकारिक पादप-स्वास्थ्य नैदानिक रिपोर्ट डाउनलोड करें (PDF)",
        "diagnosis_confirmed": "नैदानिक पुष्टि संपन्न",
        "select_sample_prompt": "👈 रोग निदान शुरू करने के लिए कृपया नमूना पत्ती चुनें या अपनी पत्ती की तस्वीर अपलोड करें।",
        "supported_crops_heading": "समर्थित फसलें एवं रोग (38 श्रेणियां):",
        "none_upload_own": "कोई नहीं (अपनी पत्ती की फोटो अपलोड करें)",
        "sample_prefix": "नमूना",
        "agro_open_review_btn": "🩺 नैदानिक समीक्षा एवं प्रमाणन पैनल खोलें ➔",
        "farmer_open_spray_btn": "🚜 14 दिवसीय छिड़काव कैलेंडर और खुराक कैलकुलेटर खोलें ➔",
        "report_id_label": "रिपोर्ट आईडी",
        "location_label": "स्थान",
        "model_core_label": "मॉडल: विज़न मांबा SSM (224px)",
    },
    "mr": {
        "title": "🌿 प्लांट मांबा: पीक आरोग्यासाठी प्रगत व्हिज्युअल इंटेलिजन्स",
        "sub_title": "व्हिजन मांबा (ResNet-18 + द्वि-दिशात्मक SSM), Grad-CAM आणि कृषी सल्लागार प्रणालीद्वारे समर्थित",
        "quality_passed": "प्रतिमेचे रिझोल्यूशन, फोकस आणि प्रकाश नैदानिक निकषांनुसार अचूक आहे.",
        "quality_warning": "प्रतिमेची गुणवत्ता कमी आहे. अस्पष्टता किंवा प्रकाशामुळे निदानात फरक पडू शकतो.",
        "predicted_disease": "निदान झालेला रोग",
        "confidence_score": "अचूकता गुण",
        "severity_level": "तीव्रता पातळी",
        "infected_leaf_area": "बाधित पानाचे क्षेत्र",
        "weather_context": "⛅ हवामान संदर्भ आणि रोग जोखीम",
        "similar_cases": "🔍 ऐतिहासिक जुने नैदानिक संदर्भ",
        "treatment_recommendations": "💊 शिफारस केलेले उपचार व व्यवस्थापन",
        "chem_bio_treatment": "रासायनिक / जैविक फवारणी",
        "cultural_management": "मशागत व सेंद्रिय प्रतिबंधात्मक उपाय",
        "clinical_note": "रोगाच्या तीव्रतेविषयी टीप",
        "xai_header": "🔍 व्हिज्युअल स्पष्टीकरणात्मक एआय (Grad-CAM) व बाधित भाग",
        "original_photo": "मूळ पानाचा फोटो",
        "attention_heatmap": "ग्रॅड-कॅम उष्णता आलेख (Heatmap)",
        "heatmap_caption": "लाल आणि पिवळे भाग रोगाचा प्रादुर्भाव दर्शवतात ज्यावर मॉडेलने निर्णय घेतला आहे.",
        "top_alternatives": "📊 इतर 3 संभाव्य रोगांची टक्केवारी",
        "download_pdf_btn": "📄 सविस्तर निदान अहवाल डाउनलोड करा (PDF)",
        "generating_pdf": "कृषी निदान पीडीएफ अहवाल तयार होत आहे...",
        "language_select": "भाषा निवडा",
        "region_select": "तुमचा शेती विभाग निवडा",
        "upload_label": "पानाचा फोटो अपलोड करा (JPG/PNG)",
        "sample_select_label": "चाचणीसाठी पानाचा नमुना निवडा:",
        "risk_level_label": "जोखीम पातळी",
        "match_label": "तंतोतंत",
        "region_label": "विभाग",
        "nav_heading": "नेव्हिगेशन",
        "new_diagnosis": "➕ नवीन निदान",
        "farm_location_weather": "शेताचे ठिकाण व हवामान",
        "recent_consultations": "मागील तपासणी इतिहास",
        "all_field_submissions": "सर्व शेतातील नोंदी",
        "clear_history": "सर्व इतिहास हटवा",
        "no_consultations": "कोणतीही मागील नोंद नाही. वरील बटणावरून पहिली तपासणी करा!",
        "logout": "🚪 बाहेर पडा",
        "logged_in_as": "वापरकर्ता:",
        "chief_agronomist_tag": "🩺 मुख्य कृषी तज्ज्ञ • नैदानिक तपासणी प्राधिकरण",
        "farmer_tag": "🚜 प्रगतशील शेतकरी • पीक संरक्षण व फवारणी मित्र",
        "nav_scanner": "🌿 एआय पीक रोग स्कॅनर",
        "nav_review": "🩺 नैदानिक तपासणी व प्रमाणीकरण",
        "nav_spray_calendar": "🚜 फवारणी वेळापत्रक आणि औषध प्रमाण कॅल्क्युलेटर",
        "confidence_label": "निदान अचूकता",
        "severity_label": "रोगाची तीव्रता",
        "infected_area_label": "बाधित पानाचे क्षेत्र",
        "microclimate_risk": "हवामान जोखीम",
        "treatment_heading": "रासायनिक व जैविक फवारणी",
        "prevention_heading": "मशागत, स्वच्छता व प्रतिबंधात्मक काळजी",
        "historical_heading": "विभागीय ऐतिहासिक रोग संदर्भ",
        "submitted_leaf": "🍃 तपासणीसाठी दिलेले पान",
        "gradcam_heatmap": "🔥 ग्रॅड-कॅम रोगट भाग उष्णता आलेख",
        "download_pdf": "📄 अधिकृत कृषी आरोग्य तपासणी अहवाल डाउनलोड करा (PDF)",
        "diagnosis_confirmed": "नैदानिक पुष्टीकरण पूर्ण",
        "select_sample_prompt": "👈 रोग निदान सुरू करण्यासाठी कृपया पानाचा नमुना निवडा किंवा तुमच्या शेतातील पानाचा फोटो अपलोड करा.",
        "supported_crops_heading": "तपासणीस उपलब्ध पिके व रोग (३८ प्रकार):",
        "none_upload_own": "काही नाही (स्वतःच्या पानाचा फोटो अपलोड करा)",
        "sample_prefix": "नमुना",
        "agro_open_review_btn": "🩺 नैदानिक तपासणी व प्रमाणीकरण पॅनेल उघडा ➔",
        "farmer_open_spray_btn": "🚜 १४ दिवसांचे फवारणी वेळापत्रक आणि औषध प्रमाण काढा ➔",
        "report_id_label": "अहवाल आयडी",
        "location_label": "ठिकाण",
        "model_core_label": "मॉडेल: व्हिजन मांबा SSM (224px)",
    }
}

DISEASE_TRANSLATIONS = {
    'Apple___Apple_scab': {'hi': 'सेब का स्कैब रोग', 'mr': 'सफरचंदाचा स्कॅब रोग'},
    'Apple___Black_rot': {'hi': 'सेब का ब्लैक रॉट (काला सड़न)', 'mr': 'सफरचंदाचा ब्लॅक रॉट (काळी कुज)'},
    'Apple___Cedar_apple_rust': {'hi': 'सेब का सीडर रस्ट (रतुआ रोग)', 'mr': 'सफरचंदाचा सीडर तांबेरा रोग'},
    'Apple___healthy': {'hi': 'सेब (स्वस्थ पत्ता)', 'mr': 'सफरचंद (निरोगी पान)'},
    'Blueberry___healthy': {'hi': 'ब्लूबेरी (स्वस्थ पत्ता)', 'mr': 'ब्लूबेरी (निरोगी पान)'},
    'Cherry_(including_sour)___Powdery_mildew': {'hi': 'चेरी पाउडरी मिल्ड्यू (भूरी)', 'mr': 'चेरीवरील भुरी रोग'},
    'Cherry_(including_sour)___healthy': {'hi': 'चेरी (स्वस्थ पत्ता)', 'mr': 'चेरी (निरोगी पान)'},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'hi': 'मक्का धूसर पत्ती धब्बा (ग्रे लीफ स्पॉट)', 'mr': 'मक्यावरील राखाडी ठिपके'},
    'Corn_(maize)___Common_rust_': {'hi': 'मक्के का सामान्य रतुआ (गेरुआ)', 'mr': 'मक्यावरील तांबेरा रोग'},
    'Corn_(maize)___Northern_Leaf_Blight': {'hi': 'मक्का उत्तरी पत्ती झुलसा', 'mr': 'मक्यावरील उत्तरेकडील करपा'},
    'Corn_(maize)___healthy': {'hi': 'मक्का (स्वस्थ पत्ता)', 'mr': 'मका (निरोगी पान)'},
    'Grape___Black_rot': {'hi': 'अंगूर का ब्लैक रॉट', 'mr': 'द्राक्षांचा काळा कुजवा रोग'},
    'Grape___Esca_(Black_Measles)': {'hi': 'अंगूर का एस्का (ब्लैक मीसल्स)', 'mr': 'द्राक्षांवरील एस्का रोग'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'hi': 'अंगूर पत्ती झुलसा', 'mr': 'द्राक्षांवरील पानांचा करपा'},
    'Grape___healthy': {'hi': 'अंगूर (स्वस्थ पत्ता)', 'mr': 'द्राक्षे (निरोगी पान)'},
    'Orange___Haunglongbing_(Citrus_greening)': {'hi': 'संतरा साइट्रस ग्रीनिंग (हुआंगलोंगबिंग)', 'mr': 'संत्र्यावरील सिट्रस ग्रीनिंग'},
    'Peach___Bacterial_spot': {'hi': 'आड़ू का जीवाणु धब्बा (बैक्टीरियल स्पॉट)', 'mr': 'पिचवरील जिवाणूजन्य ठिपके'},
    'Peach___healthy': {'hi': 'आड़ू (स्वस्थ पत्ता)', 'mr': 'पिच (निरोगी पान)'},
    'Pepper,_bell___Bacterial_spot': {'hi': 'शिमला मिर्च का जीवाणु धब्बा', 'mr': 'ढोबळी मिरचीवरील जिवाणूजन्य ठिपके'},
    'Pepper,_bell___healthy': {'hi': 'शिमला मिर्च (स्वस्थ पत्ता)', 'mr': 'ढोबळी मिरची (निरोगी पान)'},
    'Potato___Early_blight': {'hi': 'आलू का अगेती झुलसा', 'mr': 'बटाट्यावरील लवकर येणारा करपा'},
    'Potato___Late_blight': {'hi': 'आलू का पछेती झुलसा', 'mr': 'बटाट्यावरील उशिरा येणारा करपा'},
    'Potato___healthy': {'hi': 'आलू (स्वस्थ पत्ता)', 'mr': 'बटाटा (निरोगी पान)'},
    'Raspberry___healthy': {'hi': 'रास्पबेरी (स्वस्थ पत्ता)', 'mr': 'रास्पबेरी (निरोगी पान)'},
    'Soybean___healthy': {'hi': 'सोयाबीन (स्वस्थ पत्ता)', 'mr': 'सोयाबीन (निरोगी पान)'},
    'Squash___Powdery_mildew': {'hi': 'कद्दूवर्गीय पाउडरी मिल्ड्यू (भूरी रोग)', 'mr': 'भोपळा वर्गीय भुरी रोग'},
    'Strawberry___Leaf_scorch': {'hi': 'स्ट्रॉबेरी पत्ती झुलसा (लीफ स्कॉर्च)', 'mr': 'स्ट्रॉबेरीवरील पानांचा करपा'},
    'Strawberry___healthy': {'hi': 'स्ट्रॉबेरी (स्वस्थ पत्ता)', 'mr': 'स्ट्रॉबेरी (निरोगी पान)'},
    'Tomato___Bacterial_spot': {'hi': 'टमाटर का जीवाणु धब्बा', 'mr': 'टोमॅटोवरील जिवाणूजन्य ठिपके'},
    'Tomato___Early_blight': {'hi': 'टमाटर का अगेती झुलसा', 'mr': 'टोमॅटोवरील लवकर येणारा करपा'},
    'Tomato___Late_blight': {'hi': 'टमाटर का पछेती झुलसा', 'mr': 'टोमॅटोवरील उशिरा येणारा करपा'},
    'Tomato___Leaf_Mold': {'hi': 'टमाटर की पत्ती का फफूंद (लीफ मोल्ड)', 'mr': 'टोमॅटोवरील पानावरील बुरशी'},
    'Tomato___Septoria_leaf_spot': {'hi': 'टमाटर सेप्टोरिया पत्ती धब्बा', 'mr': 'टोमॅटोवरील सेप्टोरिया ठिपके'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'hi': 'टमाटर लाल मकड़ी (स्पाइडर माइट्स)', 'mr': 'टोमॅटोवरील लाल कोळी'},
    'Tomato___Target_Spot': {'hi': 'टमाटर का टारगेट स्पॉट', 'mr': 'टोमॅटोवरील टार्गेट स्पॉट'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'hi': 'टमाटर पीली पत्ती मुड़न वायरस (पर्णकुंचन)', 'mr': 'टोमॅटोवरील पर्णगुच्छ रोग'},
    'Tomato___Tomato_mosaic_virus': {'hi': 'टमाटर मोजेक वायरस', 'mr': 'टोमॅटोवरील मोझॅक व्हायरस'},
    'Tomato___healthy': {'hi': 'टमाटर (स्वस्थ पत्ता)', 'mr': 'टोमॅटो (निरोगी पान)'}
}

# Add clean names mapping
for raw_name, translations in list(DISEASE_TRANSLATIONS.items()):
    clean = CLEAN_CLASS_NAMES.get(raw_name)
    if clean:
        DISEASE_TRANSLATIONS[clean] = translations


def get_text(key, lang="en"):
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return lang_dict.get(key, TRANSLATIONS["en"].get(key, key))


def get_localized_disease_name(disease_name, lang="en"):
    clean_name = CLEAN_CLASS_NAMES.get(disease_name, disease_name)
    if lang == "en":
        return clean_name
    trans = DISEASE_TRANSLATIONS.get(disease_name, DISEASE_TRANSLATIONS.get(clean_name, {}))
    return trans.get(lang, clean_name)


def get_localized_severity(severity_str, lang="en"):
    """
    Translates severity level ratings into Hindi or Marathi.
    """
    if lang == "hi":
        if "Healthy" in severity_str or "None" in severity_str:
            return "पूर्णतः स्वस्थ / कोई संक्रमण नहीं (< 2%)"
        elif "Mild" in severity_str:
            return "हल्का संक्रमण (2% - 10%)"
        elif "Moderate" in severity_str:
            return "मध्यम संक्रमण (10% - 25%)"
        elif "Severe" in severity_str:
            return "गंभीर संक्रमण (> 25%)"
    elif lang == "mr":
        if "Healthy" in severity_str or "None" in severity_str:
            return "पूर्णपणे निरोगी / संसर्ग नाही (< 2%)"
        elif "Mild" in severity_str:
            return "सौम्य संसर्ग (2% - 10%)"
        elif "Moderate" in severity_str:
            return "मध्यम संसर्ग (10% - 25%)"
        elif "Severe" in severity_str:
            return "तीव्र / गंभीर संसर्ग (> 25%)"
    return severity_str


def get_localized_treatment_fields(treatment_dict, lang="en"):
    """
    Extracts the localized treatment, prevention, and severity note strings.
    """
    if not isinstance(treatment_dict, dict):
        return {"treatment": str(treatment_dict), "prevention": "", "severity_note": ""}

    if lang == "hi":
        t = treatment_dict.get("treatment_hi") or treatment_dict.get("treatment", "")
        p = treatment_dict.get("prevention_hi") or treatment_dict.get("prevention", "")
        s = treatment_dict.get("severity_note_hi") or treatment_dict.get("severity_note", "")
    elif lang == "mr":
        t = treatment_dict.get("treatment_mr") or treatment_dict.get("treatment", "")
        p = treatment_dict.get("prevention_mr") or treatment_dict.get("prevention", "")
        s = treatment_dict.get("severity_note_mr") or treatment_dict.get("severity_note", "")
    else:
        t = treatment_dict.get("treatment", "")
        p = treatment_dict.get("prevention", "")
        s = treatment_dict.get("severity_note", "")

    return {
        "treatment": t,
        "prevention": p,
        "severity_note": s,
        "crop": treatment_dict.get("crop", ""),
        "type": treatment_dict.get("type", "")
    }
