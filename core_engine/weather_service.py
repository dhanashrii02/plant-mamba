"""
Weather & Agronomic Risk Analysis Service
Fetches real-time location weather (temperature, humidity) and calculates
pathogen transmission risk based on current microclimate.
"""

import requests

POPULAR_REGIONS = {
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nashik": {"lat": 19.9975, "lon": 73.7898},
    "Aurangabad (Chh. Sambhajinagar)": {"lat": 19.8762, "lon": 75.3433},
    "Amravati": {"lat": 20.9374, "lon": 77.7796},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433},
}


def get_live_weather(selected_city="Auto-Detect"):
    """
    Fetches real-time weather data. Uses IP geolocation if selected_city is Auto-Detect.
    """
    city = "Nagpur"
    lat, lon = 21.1458, 79.0882

    if selected_city != "Auto-Detect" and selected_city in POPULAR_REGIONS:
        city = selected_city
        lat = POPULAR_REGIONS[selected_city]["lat"]
        lon = POPULAR_REGIONS[selected_city]["lon"]
    else:
        try:
            res = requests.get("http://ip-api.com/json/", timeout=3.5).json()
            if res.get("status") == "success":
                city = res.get("city", "Nagpur")
                lat = res.get("lat", 21.1458)
                lon = res.get("lon", 79.0882)
        except Exception:
            city = "Nagpur"

    temp = 25.0
    humidity = 85.0
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        w = requests.get(url, timeout=3.5).json()
        curr = w.get("current", {})
        temp = curr.get("temperature_2m", 25.0)
        humidity = curr.get("relative_humidity_2m", 85.0)
    except Exception:
        pass

    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "temperature": round(temp, 1),
        "humidity": round(humidity, 1)
    }


def calculate_disease_risk(disease_or_weather, humidity_or_disease=None, temperature=None, lang="en"):
    """
    Evaluates microclimate risk for the given disease with full trilingual support.
    Supports both signatures:
    1. calculate_disease_risk(disease_name, humidity, temperature, lang="en")
    2. calculate_disease_risk(weather_dict, disease_name, lang="en")
    """
    if isinstance(disease_or_weather, dict):
        weather = disease_or_weather
        disease_name = str(humidity_or_disease) if humidity_or_disease is not None else "Unknown"
        humidity = float(weather.get("humidity", 75.0))
        temp = float(weather.get("temperature", 25.0))
        if isinstance(temperature, str):
            lang = temperature
        temperature = temp
    else:
        disease_name = str(disease_or_weather)
        humidity = float(humidity_or_disease) if humidity_or_disease is not None else 75.0
        temperature = float(temperature) if temperature is not None else 25.0

    from multilingual import get_localized_disease_name
    dis_loc = get_localized_disease_name(disease_name, lang)
    is_healthy = "healthy" in disease_name.lower()

    if is_healthy:
        if lang == "hi":
            level = "फसल विकास के लिए अनुकूल"
            exp = f"वर्तमान मौसम ({temperature}°C, {humidity}% RH) स्वस्थ और तनावमुक्त फसल विकास का समर्थन करता है।"
        elif lang == "mr":
            level = "पीक वाढीसाठी अत्यंत अनुकूल"
            exp = f"सध्याचे हवामान ({temperature}°C, {humidity}% RH) पिकाच्या जोमदार आणि निरोगी वाढीसाठी अत्यंत पोषक आहे."
        else:
            level = "Optimal Growing Conditions"
            exp = f"Current weather ({temperature}°C, {humidity}% RH) in your region supports robust, unstressed crop development."
        return {
            "risk_level": level,
            "risk_color": "#10b981", # green
            "explanation": exp
        }

    # Fungal diseases: Blight, Rust, Rot, Mold, Mildew, Scab
    fungal_keywords = ["Blight", "Rust", "Rot", "Mold", "Mildew", "Scab"]
    is_fungal = any(k.lower() in disease_name.lower() for k in fungal_keywords)

    if is_fungal:
        if humidity >= 80:
            if lang == "hi":
                level = "अति-गंभीर जोखिम (Critical Risk)"
                exp = f"उच्च आर्द्रता ({humidity}%) {dis_loc} के फंगल बीजाणु अंकुरण और घावों के फैलाव को तेजी से बढ़ावा देती है।"
            elif lang == "mr":
                level = "अति-गंभीर जोखीम (Critical Risk)"
                exp = f"हवेतील जास्त दमटपणा ({humidity}%) {dis_loc} चे बीजाणू वाढण्यास व प्रादुर्भाव वेगाने पसरण्यास पोषक आहे."
            else:
                level = "Critical Risk"
                exp = f"High humidity ({humidity}%) actively accelerates spore germination and secondary lesion expansion for {dis_loc}."
            return {
                "risk_level": level,
                "risk_color": "#ef4444", # red
                "explanation": exp
            }
        elif humidity >= 65:
            if lang == "hi":
                level = "बढ़ा हुआ जोखिम (Elevated Risk)"
                exp = f"मध्यम-उच्च आर्द्रता ({humidity}%) फंगल संक्रमण और कैनोपी में बीमारी पनपने के लिए अनुकूल नमी प्रदान करती है।"
            elif lang == "mr":
                level = "वाढलेली जोखीम (Elevated Risk)"
                exp = f"मध्यम ते उच्च आर्द्रता ({humidity}%) पानांवर बुरशीची वाढ होण्यासाठी आणि संसर्ग वाढण्यासाठी पोषक आहे."
            else:
                level = "Elevated Risk"
                exp = f"Moderate-high humidity ({humidity}%) provides favorable moisture for fungal canopy incubation."
            return {
                "risk_level": level,
                "risk_color": "#f59e0b", # amber
                "explanation": exp
            }
        else:
            if lang == "hi":
                level = "कम वातावरणीय जोखिम (Low Risk)"
                exp = f"शुष्क हवा ({humidity}% RH) हवा में बीजाणुओं के प्रसार को रोकती है, फिर भी कैनोपी की नियमित जांच करें।"
            elif lang == "mr":
                level = "कमी वातावरणीय जोखीम (Low Risk)"
                exp = f"कोरडी हवा ({humidity}% RH) हवेमार्फत बुरशी पसरण्यास अटकाव करते, तरीही झाडाच्या खालच्या भागावर लक्ष ठेवावे."
            else:
                level = "Low Atmospheric Risk"
                exp = f"Dry air ({humidity}% RH) inhibits aerial spore dispersal, though localized canopy humidity should be managed."
            return {
                "risk_level": level,
                "risk_color": "#3b82f6", # blue
                "explanation": exp
            }
    else:
        # Viral / Bacterial
        if temperature >= 28 and humidity < 60:
            if lang == "hi":
                level = "उच्च कीट/वाहक जोखिम (High Vector Risk)"
                exp = f"गर्म व शुष्क मौसम ({temperature}°C) कीटों (सफेद मक्खी/माहू) की सक्रियता को बढ़ाता है जो {dis_loc} फैलाते हैं।"
            elif lang == "mr":
                level = "उच्च कीड/वाहक जोखीम (High Vector Risk)"
                exp = f"उष्ण व कोरडे वातावरण ({temperature}°C) रसशोषक किडींची संख्या वाढवते ज्यामुळे {dis_loc} चा प्रसार होतो."
            else:
                level = "High Vector Risk"
                exp = f"Warm, dry microclimate ({temperature}°C) accelerates insect vector activity transmitting {dis_loc}."
            return {
                "risk_level": level,
                "risk_color": "#f97316", # orange
                "explanation": exp
            }
        else:
            if lang == "hi":
                level = "मध्यम जोखिम (Moderate Risk)"
                exp = f"वर्तमान तापमान ({temperature}°C) और आर्द्रता ({humidity}%) में खेत की सामान्य स्वच्छता और निगरानी आवश्यक है।"
            elif lang == "mr":
                level = "मध्यम जोखीम (Moderate Risk)"
                exp = f"सध्याचे तापमान ({temperature}°C) आणि आर्द्रता ({humidity}%) पाहता बागेची नियमित स्वच्छता व पाहणी आवश्यक आहे."
            else:
                level = "Moderate Risk"
                exp = f"Current regional temperature ({temperature}°C) and humidity ({humidity}%) require routine field sanitation."
            return {
                "risk_level": level,
                "risk_color": "#f59e0b",
                "explanation": exp
            }
