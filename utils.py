import streamlit as st
import json
import os
from pathlib import Path
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Constants ---
MANIFEST_PATH = Path(__file__).parent / "content_manifest.json"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index_google_v1"
GOOGLE_EMBEDDING_MODEL = 'models/embedding-001'
COURSE_CONTENT_ROOT = Path(__file__).parent / "course-content"
NUM_FETCH_DOCS = 20  # Number of documents to fetch initially for MMR
NUM_FINAL_DOCS = 8  # Number of documents to select using MMR for the context

# Initialize Google API
def initialize_google_api():
    """Initialize the Google API with the API key."""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=google_api_key)
        return True
    except KeyError:
        st.error("GOOGLE_API_KEY not found in Streamlit secrets or .env file. Please ensure it is set.")
        return False
    except Exception as e:
        st.error(f"Error initializing Google GenAI SDK: {e}. Please ensure GOOGLE_API_KEY is set.")
        return False

# Load FAISS index
def load_faiss_index():
    """Load the FAISS index if it exists."""
    if FAISS_INDEX_PATH.exists():
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
            faiss_index = FAISS.load_local(
                str(FAISS_INDEX_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
            return faiss_index
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            return None
    else:
        st.error(f"FAISS index not found at {FAISS_INDEX_PATH}")
        return None

# Load manifest
def load_manifest():
    """Load the content manifest file."""
    try:
        if MANIFEST_PATH.exists():
            with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            
            import pandas as pd
            df = pd.DataFrame(manifest_data)
            return df
        else:
            st.error(f"Manifest file not found at {MANIFEST_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading manifest: {e}")
        return None

# Get Gemini model
def get_gemini_model(model_name="gemini-2.0-flash-exp"):
    """Get the Gemini model."""
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error configuring Google Gemini: {e}")
        return None
