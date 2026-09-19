"""
Weather & Agronomic Risk Analysis Service
Fetches real-time location weather (temperature, humidity) and calculates
pathogen transmission risk based on current microclimate.
Supports browser GPS coordinates, client IP lookup, and blocks foreign cloud datacenters.
"""

import requests

POPULAR_REGIONS = {
    # Maharashtra Key Agricultural Districts
    "Nashik": {"lat": 19.9975, "lon": 73.7898, "state": "Maharashtra", "desc": "Grapes, Onions & Vegetables"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "desc": "Floriculture & Horticulture"},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra", "desc": "Oranges & Cotton"},
    "Chh. Sambhajinagar (Aurangabad)": {"lat": 19.8762, "lon": 75.3433, "state": "Maharashtra", "desc": "Cotton, Maize & Pulses"},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433, "state": "Maharashtra", "desc": "Sugarcane & Jaggery"},
    "Solapur": {"lat": 17.6599, "lon": 75.9064, "state": "Maharashtra", "desc": "Pomegranate & Sorghum"},
    "Ahmednagar": {"lat": 19.0948, "lon": 74.7480, "state": "Maharashtra", "desc": "Sugarcane & Vegetables"},
    "Satara": {"lat": 17.6805, "lon": 74.0183, "state": "Maharashtra", "desc": "Strawberry & Turmeric"},
    "Sangli": {"lat": 16.8524, "lon": 74.5815, "state": "Maharashtra", "desc": "Turmeric, Grapes & Raisins"},
    "Jalgaon": {"lat": 21.0077, "lon": 75.5626, "state": "Maharashtra", "desc": "Banana Capital & Cotton"},
    "Amravati": {"lat": 20.9374, "lon": 77.7796, "state": "Maharashtra", "desc": "Oranges & Soybean"},
    "Akola": {"lat": 20.7002, "lon": 77.0082, "state": "Maharashtra", "desc": "Pulses & Cotton Research"},
    "Latur": {"lat": 18.4088, "lon": 76.5604, "state": "Maharashtra", "desc": "Soybean & Oilseeds Hub"},
    "Nanded": {"lat": 19.1383, "lon": 77.3210, "state": "Maharashtra", "desc": "Banana & Cotton"},
    "Yavatmal": {"lat": 20.3888, "lon": 78.1204, "state": "Maharashtra", "desc": "Cotton Belt"},
    "Dhule": {"lat": 20.9042, "lon": 74.7749, "state": "Maharashtra", "desc": "Chili & Bajra"},
    "Ratnagiri": {"lat": 16.9902, "lon": 73.3120, "state": "Maharashtra", "desc": "Alphonso Mango & Cashew"},
    "Mumbai / Thane": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "desc": "Coastal Agro Market"},
    
    # National Agricultural Hubs
    "Indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh", "desc": "Wheat & Soybean"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "desc": "Pulses & Gram"},
    "Chandigarh / Ludhiana": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab", "desc": "Wheat & Rice Bowl"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "desc": "Cotton, Chili & Rice"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "desc": "Horticulture & Coffee"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "desc": "Mango, Potato & Sugarcane"},
    "New Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi NCR", "desc": "IARI Central Research"},
}


def get_live_weather(selected_city="Auto-Detect", client_ip=None, gps_coords=None):
    """
    Fetches real-time weather data.
    Priority:
    1. Exact Browser GPS coordinates if provided (lat, lon)
    2. Selected city from POPULAR_REGIONS
    3. Client IP geolocation (using client's public IP from request headers)
    4. Safe fallback to Nashik (Maharashtra Agro Hub) if cloud datacenter (e.g. US / The Dalles) is detected.
    """
    DATACENTER_CITIES = {
        "the dalles", "council bluffs", "north bergen", "ashburn", "boardman",
        "mountain view", "santa clara", "seattle", "des moines", "quincy"
    }

    city = "Nashik"
    lat, lon = 19.9975, 73.7898
    source_tag = "Default"

    # Case 1: Exact GPS coordinates from client browser
    if gps_coords and isinstance(gps_coords, dict) and "lat" in gps_coords and "lon" in gps_coords:
        try:
            g_lat = float(gps_coords["lat"])
            g_lon = float(gps_coords["lon"])
            lat, lon = g_lat, g_lon
            source_tag = "Live GPS"
            try:
                rg_url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
                rg = requests.get(rg_url, timeout=2.5).json()
                loc = rg.get("city") or rg.get("locality") or rg.get("principalSubdivision")
                if loc:
                    city = loc
                else:
                    city = f"GPS ({lat:.2f}°, {lon:.2f}°)"
            except Exception:
                city = f"GPS ({lat:.2f}°, {lon:.2f}°)"
        except Exception:
            pass

    # Case 2: Selected from POPULAR_REGIONS
    elif selected_city != "Auto-Detect" and selected_city in POPULAR_REGIONS:
        city = selected_city
        lat = POPULAR_REGIONS[selected_city]["lat"]
        lon = POPULAR_REGIONS[selected_city]["lon"]
        source_tag = "Selected"

    # Case 3: Auto-Detect via IP
    else:
        detected = False
        # Try client IP if provided and public
        if client_ip and isinstance(client_ip, str) and not client_ip.startswith(("127.", "10.", "192.168.", "172.")):
            try:
                res = requests.get(f"http://ip-api.com/json/{client_ip}", timeout=2.5).json()
                if res.get("status") == "success":
                    det_city = res.get("city", "")
                    country_code = res.get("countryCode", "")
                    if country_code == "IN" or (res.get("country") == "India" and det_city.lower() not in DATACENTER_CITIES):
                        city = det_city
                        lat = res.get("lat", 19.9975)
                        lon = res.get("lon", 73.7898)
                        detected = True
                        source_tag = "Client IP"
            except Exception:
                pass

        if not detected:
            try:
                res = requests.get("http://ip-api.com/json/", timeout=2.5).json()
                if res.get("status") == "success":
                    det_city = res.get("city", "")
                    country_code = res.get("countryCode", "")
                    if country_code == "IN" and det_city.lower() not in DATACENTER_CITIES:
                        city = det_city
                        lat = res.get("lat", 19.9975)
                        lon = res.get("lon", 73.7898)
                        detected = True
                        source_tag = "Auto IP"
            except Exception:
                pass

        if not detected:
            city = "Nashik (Agro Hub)"
            lat, lon = 19.9975, 73.7898
            source_tag = "Agro Hub"

    temp = 26.5
    humidity = 68.0
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        w = requests.get(url, timeout=3.0).json()
        curr = w.get("current", {})
        temp = curr.get("temperature_2m", 26.5)
        humidity = curr.get("relative_humidity_2m", 68.0)
    except Exception:
        pass

    return {
        "city": city,
        "lat": lat,
        "lon": lon,
        "temperature": round(temp, 1),
        "humidity": round(humidity, 1),
        "source": source_tag
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
