def construction_risk(soil_type, moisture, temp, humidity, rainfall):
    risks = {}

    # Settlement Risk
    if moisture < 35 and temp > 32:
        risks["Settlement"] = ("High", "Dry soil shrinkage → foundation movement risk")
    elif rainfall > 80:
        risks["Settlement"] = ("Moderate", "Wet soil → compression risk")
    else:
        risks["Settlement"] = ("Low", "Stable condition")

    # Bearing Capacity
    if moisture > 70:
        risks["Bearing Capacity"] = ("Low", "Too wet → weak support")
    elif soil_type.lower() in ["sandy", "gravel"]:
        risks["Bearing Capacity"] = ("High", "Good support")
    else:
        risks["Bearing Capacity"] = ("Moderate", "Average support")

    # Erosion
    if rainfall > 100:
        risks["Erosion"] = ("High", "Heavy rain → soil loss")
    elif rainfall > 50:
        risks["Erosion"] = ("Moderate", "Possible erosion")
    else:
        risks["Erosion"] = ("Low", "Safe")

    # Shrink-Swell
    if soil_type.lower() == "clay":
        if moisture < 30 or moisture > 75:
            risks["Shrink-Swell"] = ("High", "Crack risk")
        else:
            risks["Shrink-Swell"] = ("Moderate", "Manageable")
    else:
        risks["Shrink-Swell"] = ("Low", "Not significant")

    return risks