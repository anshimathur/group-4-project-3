import streamlit as st
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import our utils module
sys.path.append(str(Path(__file__).parent.parent))
import utils

# Page configuration
st.set_page_config(layout="wide", page_title="Flashcards", page_icon="🔄")

st.title("AI Course Tutor - Flashcards")
st.write("Generate flashcards on course topics to help you study.")

# Initialize Google API
api_initialized = utils.initialize_google_api()
if not api_initialized:
    st.stop()

# Load FAISS index
faiss_index = utils.load_faiss_index()
if faiss_index is None:
    st.error("Failed to load the FAISS index. Flashcard generation is disabled.")
    st.stop()

# Get Gemini model
gemini_model = utils.get_gemini_model()
if gemini_model is None:
    st.error("Failed to initialize the Gemini model. Flashcard generation is disabled.")
    st.stop()

# Initialize session state for flashcards
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
    
if "current_card_index" not in st.session_state:
    st.session_state.current_card_index = 0
    
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False

# Function to generate flashcards
def generate_flashcards(topic, num_cards=5):
    try:
        # Retrieve relevant content about the topic
        docs = faiss_index.similarity_search(topic, k=3)
        
        if not docs:
            return [], "No relevant content found for this topic."
            
        # Prepare context from the retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt for flashcard generation
        prompt = f"""
        Based *strictly* on the following course content about '{topic}', create {num_cards} high-quality flashcards.
        
        COURSE CONTENT:
        {context}
        
        INSTRUCTIONS:
        1. Each flashcard should have a clear question on one side and a concise answer on the other.
        2. Questions should test understanding, not just memorization.
        3. Format your response as a JSON array of objects, each with 'question' and 'answer' keys.
        4. Make the flashcards challenging but fair, focusing on key concepts.
        5. Each flashcard should be independent and self-contained.
        
        RESPONSE FORMAT:
        [
            {{"question": "...", "answer": "..."}},
            {{"question": "...", "answer": "..."}},
            ...
        ]
        """
        
        # Generate flashcards using Gemini
        response = gemini_model.generate_content(prompt)
        
        # Process the response
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            response_text = ''.join([part.text for part in response.parts])
        else:
            return [], "Failed to generate flashcards. Invalid response format."
        
        # Extract JSON from the response text
        import json
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if json_match:
            try:
                flashcards = json.loads(json_match.group())
                return flashcards, None
            except json.JSONDecodeError:
                pass
        
        # If parsing the JSON failed, try a more lenient approach
        try:
            # Clean up the text to make it valid JSON
            # Remove markdown code block syntax
            clean_text = re.sub(r'```json|```', '', response_text).strip()
            flashcards = json.loads(clean_text)
            return flashcards, None
        except json.JSONDecodeError as e:
            return [], f"Failed to parse the generated flashcards: {e}"
            
    except Exception as e:
        return [], f"Error generating flashcards: {e}"

# UI Components
with st.sidebar:
    st.header("Generate Flashcards")
    topic = st.text_input("Enter a topic from your course materials")
    num_cards = st.slider("Number of flashcards", min_value=1, max_value=10, value=5)
    
    if st.button("Generate Flashcards", key="gen_cards"):
        with st.spinner("Generating flashcards..."):
            flashcards, error = generate_flashcards(topic, num_cards)
            if error:
                st.error(error)
            else:
                st.session_state.flashcards = flashcards
                st.session_state.current_card_index = 0
                st.session_state.show_answer = False
                st.success(f"Generated {len(flashcards)} flashcards")

# Navigation functions
def next_card():
    if st.session_state.flashcards:
        st.session_state.current_card_index = (st.session_state.current_card_index + 1) % len(st.session_state.flashcards)
        st.session_state.show_answer = False

def prev_card():
    if st.session_state.flashcards:
        st.session_state.current_card_index = (st.session_state.current_card_index - 1) % len(st.session_state.flashcards)
        st.session_state.show_answer = False

def toggle_answer():
    st.session_state.show_answer = not st.session_state.show_answer

# Display flashcards
if st.session_state.flashcards:
    # Container for the flashcard
    card_container = st.container()
    with card_container:
        # Create a card-like container
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            # Card styling
            current_card = st.session_state.flashcards[st.session_state.current_card_index]
            
            # Create a card-like appearance with custom CSS
            st.markdown("""
            <style>
            .flashcard {
                background-color: #333;
                color: #fff;
                border-radius: 10px;
                padding: 20px;
                min-height: 200px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .card-content {
                font-size: 1.2rem;
                margin: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display card content
            st.markdown(f"""
            <div class="flashcard">
                <div class="card-content">
                    <h3>Question:</h3>
                    <p>{current_card["question"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display answer if button is clicked
            if st.session_state.show_answer:
                st.markdown(f"""
                <div class="flashcard">
                    <div class="card-content">
                        <h3>Answer:</h3>
                        <p>{current_card["answer"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Navigation buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.button("⬅️ Previous", on_click=prev_card)
            with col2:
                st.button("🔄 Flip Card", on_click=toggle_answer)
            with col3:
                st.button("Next ➡️", on_click=next_card)
            
            # Card counter
            st.write(f"Card {st.session_state.current_card_index + 1} of {len(st.session_state.flashcards)}")
else:
    st.info("Enter a topic in the sidebar and click 'Generate Flashcards' to get started.")
    
    # Example topics suggestion
    st.markdown("### Suggested Topics")
    st.markdown("""
    - Machine Learning Basics
    - Neural Networks
    - Data Preprocessing
    - Model Evaluation
    - Python Programming
    - Reinforcement Learning
    """)
