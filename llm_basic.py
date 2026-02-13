import os
from google import genai

# Ensure the GEMINI_API_KEY environment variable is set
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

model_name = "gemini-2.5-flash"

response = client.models.generate_content(
    model=model_name,
    contents="Explain how the Gemini 2.5 model processes information in a simple way."
)

print(response.text)