import PIL.Image
import google.generativeai as genai
#genai.configure(api_key="AIzaSyAxhU6n6FWwQbcR6bvPcLBQX_S5YdpLYHY")
#genai.configure(api_key="AIzaSyABWxUrpQ1ycQmUIlJc6gJZTiSg5xvg6Tg")
API_KEY = "AIzaSyB_7LZ2BP0Hs5dbYrBrBvonN0_SUfZ3xUQ"


from google import genai
from google.genai import types
import json
from pydantic import BaseModel

class Plant(BaseModel):
    plant_name: str
    growing_conditions: list[str]
    advice: str

def classify_image(image_path, sensor_data):

    prompt = (
    "You are a plant expert. Based on the image of the plant I provide, "
    "please identify the plant species or the closest match. "
    "Please provide advice on how to care for this plant based on the data provided. "
    f"Temperature: {sensor_data['temperature']}°C, Humidity: {sensor_data['humidity']}%, Soil Moisture: {sensor_data['moisture']}%, Light: {sensor_data['light']} lux.\n"
    "Then give the recommended growing conditions, including:\n"
    "- Soil moisture (e.g., low/medium/high)\n"
    "- Light exposure (e.g., full sun, partial shade, indirect light)\n"
    "- Humidity levels\n"
    "- Ideal temperature range (°C or °F)\n"
    "Be concise but complete. If the plant can't be clearly identified, say so."
    )

    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    image = types.Part.from_bytes(
        data=image_bytes, mime_type="image/jpeg"
    )

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, image],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Plant],
        },
    )
    plants = json.loads(response.text)
    for plant in plants:
        print(plant['plant_name'])
        for p in plant['growing_conditions']:
            print(f"- {p}")
        print(plant['advice'])

    return plants

