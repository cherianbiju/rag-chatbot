from google import genai

# 🔴 PASTE YOUR KEY DIRECTLY HERE (just for testing)
client = genai.Client(api_key="PASTE_KEY_HERE")

print("Testing connection...")

for m in client.models.list():
    print(m.name)

print("Success")