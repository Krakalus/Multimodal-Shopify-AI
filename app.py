import streamlit as st
from transformers import pipeline
from PIL import Image
from openai import OpenAI

st.set_page_config(page_title="Snap → Shopify Gold", layout="centered")
st.title("🖼️ Snap → Shopify Gold")

# === BACKEND SWITCHER ===
backend = st.sidebar.selectbox("Backend", ["Groq (free & fast)", "OpenAI (GPT-4o)"])

if backend == "Groq (free & fast)":
    api_key = st.secrets["GROQ_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model = "llama-3.1-8b-instant"   # ← NEW KING (vision + text)
else:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    model = "gpt-4o-mini"

# === BLIP (silence warnings + use_fast) ===
@st.cache_resource
def load_captioner():
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        use_fast=True,           # ← kills the yellow warning
        device=-1                # CPU = works everywhere
    )

captioner = load_captioner()

# === UI ===
img_file = st.file_uploader("Drop sneaker / hoodie / anything", type=["png", "jpg", "jpeg"])

if img_file:
    image = Image.open(img_file)
    st.image(image, width=380)

    with st.spinner("BLIP reading the pixels..."):
        cap = captioner(image)[0]["generated_text"]

    st.write("**Caption:**", cap)

    with st.spinner("Writing Etsy fire..."):
        prompt = f"""Caption: {cap}

        Write:
        • 1 catchy Title (8 words max)
        • 2-sentence Description
        • 10 SEO tags (comma list)

        Example:
        Title: Neon Glow LED Sneakers
        Description: Hand-crafted...
        SEO Tags: led sneakers, glow shoes, ..."""

        chat = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.8
        )
        listing = chat.choices[0].message.content

    st.success("Shipped!")
    st.markdown("### Etsy Gold")
    st.markdown(listing)
