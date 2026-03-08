from google import genai

client = genai.Client(api_key="AIzaSyBtaCxXVJUbHS46oDL1_TVQgNX9GCCXMyc")

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="hello world"
)

print(len(response.embeddings[0].values))