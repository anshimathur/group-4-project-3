import streamlit as st
import sys
import os
from pathlib import Path
import json
import re
import random

# Add the parent directory to the path so we can import our utils module
sys.path.append(str(Path(__file__).parent.parent))
import utils

# Page configuration
st.set_page_config(layout="wide", page_title="Quiz Generator", page_icon="📝")

st.title("AI Course Tutor - Quiz Generator")
st.write("Generate customized quizzes based on your course materials to test your knowledge.")

# Initialize Google API
api_initialized = utils.initialize_google_api()
if not api_initialized:
    st.stop()

# Load FAISS index
faiss_index = utils.load_faiss_index()
if faiss_index is None:
    st.error("Failed to load the FAISS index. Quiz generation is disabled.")
    st.stop()

# Get Gemini model
gemini_model = utils.get_gemini_model()
if gemini_model is None:
    st.error("Failed to initialize the Gemini model. Quiz generation is disabled.")
    st.stop()

# Define suggested topics
SUGGESTED_TOPICS = [
    "Machine Learning Algorithms",
    "Deep Learning",
    "Data Visualization",
    "Statistical Analysis",
    "Natural Language Processing",
    "Computer Vision"
]

# Initialize session state for quiz
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
    
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
    
if "selected_answers" not in st.session_state:
    st.session_state.selected_answers = {}
    
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
    
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
    
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = ""

# Function to generate quiz questions
def generate_quiz(topic, num_questions=5, difficulty="medium"):
    try:
        # Retrieve relevant content about the topic
        docs = faiss_index.similarity_search(topic, k=4)
        
        if not docs:
            return [], "No relevant content found for this topic."
            
        # Prepare context from the retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt for quiz generation
        prompt = f"""
        Based on the following course content about '{topic}', create {num_questions} multiple-choice quiz questions at a {difficulty} difficulty level.
        
        COURSE CONTENT:
        {context}
        
        INSTRUCTIONS:
        1. The generated questions should be relevant to the course material.
        2. Do not generate questions that ask for the user's opinion or experience.
        3. Do not generate questions that are not relevant to the course material.
        
        INSTRUCTIONS:
        1. Each question should have 4 answer choices (A, B, C, D).
        2. Only one answer should be correct.
        3. Questions should test understanding, not just memorization.
        4. Format your response as a JSON array of objects with the following structure:
           - "question": The question text
           - "choices": An array of 4 possible answers
           - "correct_answer": The index (0-3) of the correct answer
           - "explanation": A brief explanation of why the correct answer is right
        
        RESPONSE FORMAT:
        [
            {{
                "question": "...",
                "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
                "correct_answer": 0,
                "explanation": "..."
            }},
            ...
        ]
        
        Difficulty guidance:
        - Easy: Basic recall of concepts and simple applications
        - Medium: Understanding of concepts and their relationships
        - Hard: Application of multiple concepts, analysis, and critical thinking
        """
        
        # Generate quiz using Gemini
        response = gemini_model.generate_content(prompt)
        
        # Process the response
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            response_text = ''.join([part.text for part in response.parts])
        else:
            return [], "Failed to generate quiz. Invalid response format."
        
        # Extract JSON from the response text
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if json_match:
            try:
                quiz_questions = json.loads(json_match.group())
                return quiz_questions, None
            except json.JSONDecodeError:
                pass
        
        # If parsing the JSON failed, try a more lenient approach
        try:
            # Clean up the text to make it valid JSON
            clean_text = re.sub(r'```json|```', '', response_text).strip()
            quiz_questions = json.loads(clean_text)
            return quiz_questions, None
        except json.JSONDecodeError as e:
            return [], f"Failed to parse the generated quiz: {e}"
            
    except Exception as e:
        return [], f"Error generating quiz: {e}"

# UI Components for generating the quiz
with st.sidebar:
    st.header("Quiz Settings")
    
    # Topic selection
    selected_topic = st.selectbox("Select a topic", ["Custom Topic"] + SUGGESTED_TOPICS, key="topic_selector")
    
    # If "Custom Topic" is selected, show text input, otherwise use the selected topic
    if selected_topic == "Custom Topic":
        topic = st.text_input("Enter a topic from your course materials")
    else:
        topic = selected_topic
        st.info(f"Selected topic: {topic}")
    
    num_questions = st.slider("Number of questions", min_value=3, max_value=10, value=5)
    difficulty = st.select_slider("Difficulty", options=["easy", "medium", "hard"], value="medium")
    
    if st.button("Generate Quiz", key="gen_quiz"):
        with st.spinner("Generating quiz questions..."):
            quiz_questions, error = generate_quiz(topic, num_questions, difficulty)
            if error:
                st.error(error)
            else:
                st.session_state.quiz_questions = quiz_questions
                st.session_state.current_question = 0
                st.session_state.selected_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.success(f"Generated {len(quiz_questions)} questions!")

# Reset quiz function
def reset_quiz():
    st.session_state.quiz_questions = []
    st.session_state.selected_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0

# Submit quiz function
def submit_quiz():
    # Calculate score
    score = 0
    for i, question in enumerate(st.session_state.quiz_questions):
        if i in st.session_state.selected_answers:
            if st.session_state.selected_answers[i] == question["correct_answer"]:
                score += 1
    
    st.session_state.quiz_score = score
    st.session_state.quiz_submitted = True

# Navigation buttons
def next_question():
    if st.session_state.current_question < len(st.session_state.quiz_questions) - 1:
        st.session_state.current_question += 1

def prev_question():
    if st.session_state.current_question > 0:
        st.session_state.current_question -= 1

# Main quiz display
if st.session_state.quiz_questions:
    # Display the quiz
    if not st.session_state.quiz_submitted:
        # Show current question
        current_q = st.session_state.quiz_questions[st.session_state.current_question]
        
        st.markdown(f"### Question {st.session_state.current_question + 1} of {len(st.session_state.quiz_questions)}")
        st.markdown(f"**{current_q['question']}**")
        
        # Display choices as radio buttons
        selected_option = st.radio(
            "Select your answer:",
            options=current_q["choices"],
            index=st.session_state.selected_answers.get(st.session_state.current_question, None),
            key=f"q_{st.session_state.current_question}"
        )
        
        # Store the selected answer
        for i, choice in enumerate(current_q["choices"]):
            if selected_option == choice:
                st.session_state.selected_answers[st.session_state.current_question] = i
                break
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Previous", on_click=prev_question, disabled=st.session_state.current_question == 0)
        with col3:
            if st.session_state.current_question < len(st.session_state.quiz_questions) - 1:
                st.button("Next", on_click=next_question)
            else:
                # Show submit button on the last question
                st.button("Submit Quiz", on_click=submit_quiz, type="primary")
        
        # Question progress bar
        progress = (st.session_state.current_question + 1) / len(st.session_state.quiz_questions)
        st.progress(progress)
        
        # Display how many questions are answered
        num_answered = len(st.session_state.selected_answers)
        st.write(f"You've answered {num_answered} out of {len(st.session_state.quiz_questions)} questions.")
    
    else:
        # Display quiz results
        st.markdown("## Quiz Results")
        st.markdown(f"**Topic:** {topic}")
        st.markdown(f"**Score:** {st.session_state.quiz_score} out of {len(st.session_state.quiz_questions)}")
        
        percentage = (st.session_state.quiz_score / len(st.session_state.quiz_questions)) * 100
        st.markdown(f"**Percentage:** {percentage:.1f}%")
        
        # Display progress bar for score
        st.progress(percentage / 100)
        
        # Display feedback based on score
        if percentage >= 80:
            st.success("Great job! You have a strong understanding of this topic.")
        elif percentage >= 60:
            st.info("Good work! You have a decent grasp of the material but there's room for improvement.")
        else:
            st.warning("You might want to review this topic again to strengthen your understanding.")
        
        # Display detailed results
        st.markdown("### Detailed Results")
        for i, question in enumerate(st.session_state.quiz_questions):
            with st.expander(f"Question {i+1}: {question['question']}"):
                selected_idx = st.session_state.selected_answers.get(i, None)
                correct_idx = question["correct_answer"]
                
                st.markdown("**Choices:**")
                for j, choice in enumerate(question["choices"]):
                    if j == correct_idx:
                        st.markdown(f"- {choice} ✅ (Correct Answer)")
                    elif j == selected_idx and j != correct_idx:
                        st.markdown(f"- {choice} ❌ (Your Answer)")
                    else:
                        st.markdown(f"- {choice}")
                
                st.markdown(f"**Explanation:** {question['explanation']}")
        
        # Offer to create a new quiz
        if st.button("Create a New Quiz", type="primary"):
            reset_quiz()
            
else:
    st.info("Configure your quiz in the sidebar and click 'Generate Quiz' to get started.")
    
    # Display available topics
    st.markdown("### Available Topics")
    st.write("Select a topic from the dropdown in the sidebar or enter a custom topic.")
    
    # Tips for effective quizzing
    with st.expander("Tips for Effective Quizzing"):
        st.markdown("""
        - **Choose focused topics** rather than broad areas for more targeted learning
        - **Vary the difficulty** to challenge yourself appropriately
        - **Take quizzes regularly** to reinforce learning and identify knowledge gaps
        - **Review explanations** carefully for questions you get wrong
        - **Create quizzes on related topics** to build comprehensive understanding
        """)
