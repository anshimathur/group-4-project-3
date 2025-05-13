import streamlit as st
import json
from pathlib import Path
import pandas as pd
import nbformat # To read notebooks
import re
import traceback # Import for printing traceback
import time # To check manifest modification time
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language # Keep Recursive here
from langchain_community.vectorstores import FAISS # Updated import
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Import Google embeddings
from langchain.docstore.document import Document # Correct import
from collections import Counter
import google.generativeai as genai # New SDK for generative model
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Google Generative AI SDK
# Ensure GOOGLE_API_KEY is set in your .env file or Streamlit secrets
# For local development, using .env is fine. For deployment, use Streamlit secrets.
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
except KeyError:
    st.error("GOOGLE_API_KEY not found in Streamlit secrets or .env file. Please ensure it is set.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Google GenAI SDK: {e}. Please ensure GOOGLE_API_KEY is set.")
    st.stop()

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Chat", page_icon="💬")

st.title("AI Course Tutor - Chat")
st.write("Ask questions about your course material and get AI-powered answers.")

# --- Constants ---
MANIFEST_PATH = Path(__file__).parent / "content_manifest.json"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index_google_v1" # New path for Google embeddings
GOOGLE_EMBEDDING_MODEL = 'models/embedding-001' # Use the correct format for embedding model name
COURSE_CONTENT_ROOT = Path(__file__).parent / "course-content" # Define root for relative paths
TRANSCRIPTS_ROOT = Path(__file__).parent / "transcripts" # Define root for transcript files
NUM_FETCH_DOCS = 20 # Number of documents to fetch initially for MMR
NUM_FINAL_DOCS = 8 # Number of diverse documents to select using MMR for the context

# --- Gemini Configuration ---
gemini_configured = False
try:
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp") # Use stable Gemini model compatible with the current SDK
    gemini_configured = True
    print("Gemini configured successfully.")
except Exception as e:
    st.error(f"Error configuring Google Gemini: {e}")
    

# --- Content Reading Functions ---

def read_py_file(file_path):
    """Reads content from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Error reading Python file {file_path}: {e}")
        return ""

def read_md_file(file_path):
    """Reads content from a Markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read() # Keep it simple for now
    except Exception as e:
        print(f"Warning: Error reading Markdown file {file_path}: {e}")
        return ""

def read_ipynb_file(file_path):
    """Reads and cleans content from a Jupyter Notebook file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        content = []
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'markdown':
                content.append(cell.source)
            elif cell.cell_type == 'code':
                # Remove only streamlit commands, keep comments
                cleaned_source = re.sub(r'^\s*st\..*', '', cell.source, flags=re.MULTILINE)
                if cleaned_source.strip():
                    content.append(cleaned_source.strip())
        return "\n\n".join(content)
    except Exception as e:
        print(f"Warning: Error reading/parsing Notebook file {file_path}: {e}")
        return ""

def read_pdf_file(file_path):
    """
    Reads text content from a PDF file, page by page.
    Returns a list of tuples: (page_number, page_content).
    """
    pages_content = []
    try:
        doc = fitz.open(stream=file_path.read_bytes(), filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            pages_content.append((page_num, text))
        doc.close()
    except Exception as e:
        print(f"Warning: Error reading PDF file {file_path}: {e}")
    return pages_content

def read_txt_file(file_path):
    """Reads content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Error reading text file {file_path}: {e}")
        return ""
        

# --- Map file extensions to reading functions ---
READERS = {
    '.py': read_py_file,
    '.md': read_md_file,
    '.ipynb': read_ipynb_file,
    '.pdf': read_pdf_file,
    '.txt': read_txt_file
}

# --- Build or Load FAISS Index ---

@st.cache_resource # Re-enable caching
def build_or_load_index(df):
    """
    Builds a FAISS index from documents or loads it.
    Checks manifest timestamp and rebuilds if manifest is newer or index is missing.
    Includes page number metadata for PDFs.
    """
    index_file = FAISS_INDEX_PATH / "index.faiss"
    index_pkl = FAISS_INDEX_PATH / "index.pkl"
    rebuild_needed = False
    manifest_mtime = 0
    index_mtime = 0

    # Check if manifest exists first
    if not MANIFEST_PATH.exists():
        st.error(f"Manifest file not found at {MANIFEST_PATH}. Please run create_manifest.py.")
        return None

    try:
        manifest_mtime = MANIFEST_PATH.stat().st_mtime
    except FileNotFoundError:
        st.error(f"Manifest file disappeared unexpectedly at {MANIFEST_PATH}.")
        return None

    # Check if index needs rebuilding
    if not FAISS_INDEX_PATH.exists() or not index_file.exists() or not index_pkl.exists():
        print("FAISS index directory or files not found. Rebuilding...")
        rebuild_needed = True
    else:
        try:
            index_mtime = index_file.stat().st_mtime
            if manifest_mtime > index_mtime:
                print("Manifest file is newer than FAISS index. Rebuilding...")
                rebuild_needed = True
        except FileNotFoundError:
            print("Index file missing during timestamp check. Rebuilding...")
            rebuild_needed = True
        except Exception as e:
            st.error(f"Error checking index/manifest timestamp: {e}. Rebuilding index...")
            rebuild_needed = True # Force rebuild on error

    if not rebuild_needed:
        try:
            print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}...")
            embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
            faiss_index = FAISS.load_local(
                FAISS_INDEX_PATH.as_posix(),
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("FAISS index loaded successfully.")
            return faiss_index
        except Exception as e:
            st.error(f"Error loading existing FAISS index: {e}. Rebuilding...")
            rebuild_needed = True

    # --- Rebuild Logic ---
    if rebuild_needed:
        if df is None or df.empty:
            st.error("Cannot build index: Content manifest DataFrame is empty or None.")
            return None

        print("Building new FAISS index...")
        all_docs_for_faiss = []
        total_files = len(df)
        processed_count = 0
        progress_bar = st.progress(0.0, text="Initializing index build...")

        # Define the text splitter
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=150 # Adjust chunk size for code? Maybe larger?
        )

        for index, row in df.iterrows():
            processed_count += 1
            relative_path = row.get('relative_path')
            if not relative_path:
                print(f"Warning: Skipping row {index} due to missing 'relative_path'.")
                continue

            file_ext = Path(relative_path).suffix.lower()
            reader = READERS.get(file_ext)
            
            # Check if this is a transcript file and use the right directory
            is_transcript = row.get('is_transcript', False)
            if is_transcript:
                file_path_to_read = TRANSCRIPTS_ROOT / relative_path
            else:
                file_path_to_read = COURSE_CONTENT_ROOT / relative_path

            # Update progress bar
            progress_text = f"Processing: {relative_path} ({processed_count}/{total_files})"
            progress_bar.progress(processed_count / total_files, text=progress_text)

            if reader and file_path_to_read.is_file():
                base_metadata = {
                    'source': str(relative_path),
                    'file_type': str(file_ext),
                    'module': str(row.get('module', 'N/A') or 'N/A'),
                    'day': str(row.get('day', 'N/A') or 'N/A'),
                    'slideshow': str(row.get('slideshow_pdf', 'N/A') or 'N/A')
                }

                try:
                    content_or_pages = reader(file_path_to_read)

                    # Handle PDF (list of pages) vs other types (single string)
                    if file_ext == '.pdf' and isinstance(content_or_pages, list):
                        for page_num, page_content in content_or_pages:
                            if page_content:
                                page_metadata = base_metadata.copy()
                                page_metadata['page'] = page_num # Add page number
                                chunks = recursive_splitter.split_text(page_content)
                                for chunk in chunks:
                                    all_docs_for_faiss.append(Document(page_content=chunk, metadata=page_metadata))
                    elif isinstance(content_or_pages, str) and content_or_pages:
                        # Handle non-PDF content
                        if file_ext == '.py':
                            splitter = python_splitter
                        else: # Use recursive for .md, .ipynb, .txt etc.
                            splitter = recursive_splitter
                        chunks = splitter.split_text(content_or_pages)
                        for chunk in chunks:
                            all_docs_for_faiss.append(Document(page_content=chunk, metadata=base_metadata))
                    # else: content was empty or None, ignore

                except Exception as e:
                    print(f"Warning: Error processing file {file_path_to_read}: {e}")
                    traceback.print_exc() # Print traceback for detailed debugging

            elif not reader:
                print(f"Warning: No reader configured for file type {file_ext} ({relative_path})")
            elif not file_path_to_read.is_file():
                exists_status = "exists but is not a file" if file_path_to_read.exists() else "does not exist"
                print(f"Warning: Path identified as '{file_ext}', but is not a valid file: {file_path_to_read} ({exists_status})")

        progress_bar.empty() # Clear progress bar

        if not all_docs_for_faiss:
            st.error("No documents could be processed for indexing. Check file paths, readers, and content.")
            return None

        embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
        faiss_index = FAISS.from_documents(all_docs_for_faiss, embeddings)

        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        faiss_index.save_local(FAISS_INDEX_PATH.as_posix())
        print(f"FAISS index successfully built and saved to {FAISS_INDEX_PATH}")
        return faiss_index

    return None # Should not be reached

# --- Load Content Manifest ---
@st.cache_data # Re-enable caching
def load_manifest():
    """Loads the content manifest file into a pandas DataFrame."""
    if not MANIFEST_PATH.exists():
        st.error(f"Content manifest file not found: {MANIFEST_PATH}")
        return None
    try:
        df = pd.read_json(MANIFEST_PATH)
        print(f"Manifest loaded successfully: {len(df)} entries.")
        return df
    except Exception as e:
        st.error(f"Error loading manifest file {MANIFEST_PATH}: {e}")
        return None

# --- Load Manifest and Build/Load Index ---
manifest_df = load_manifest()
faiss_index = None
if manifest_df is not None:
    faiss_index = build_or_load_index(manifest_df)
else:
    st.error("Failed to load manifest. Cannot initialize FAISS index.")

# --- App UI ---
# Initialize chat history


# --- Main Chat Logic ---
# Initialize messages in session state for persistence
    
# Initialize messages list for UI display
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages when page loads
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sources" in message:
            st.markdown(message["content"])
            # Display sources if available
            if message["sources"]:
                with st.expander("View Sources"):
                    for source_key, metadata in message["sources"].items():
                        source = metadata.get('source', 'Unknown')
                        page = metadata.get('page', None)
                        if page:
                            st.caption(f"📄 {source} (Page {page})")
                        else:
                            st.caption(f"📄 {source}")
                        if 'slideshow' in metadata and metadata['slideshow']:
                            st.caption(f"  Related Slideshow: `{metadata['slideshow']}`")
        else:
            st.markdown(message["content"])
if faiss_index is not None and gemini_configured:
    if prompt := st.chat_input("Ask something about the course material..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # User message already added to session state above
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare for assistant response
        full_response_content = "Error: Response generation failed."
        sources_for_display = []
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Placeholder for streaming/final answer
            try:
                with st.spinner("Searching course material and generating answer..."):
                    # 1. Retrieve relevant documents using MMR
                    retriever = faiss_index.as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': NUM_FINAL_DOCS, 'fetch_k': NUM_FETCH_DOCS} # MMR specific
                    )

                    # sources = retriever.get_relevant_documents(prompt) # DEPRECATED
                    sources = retriever.invoke(prompt) # Use invoke instead

                    # --- Prepare Context --- 
                    context = "\n\n".join([doc.page_content for doc in sources])

                    # 2. Check if results were found
                    if not sources:
                        # Check if query is about common ML/data science topics
                        common_ml_topics = ['pandas', 'numpy', 'matplotlib', 'tensorflow', 'pytorch', 'scikit-learn', 
                                        'machine learning', 'neural network', 'deep learning', 'data science', 
                                        'regression', 'classification', 'clustering', 'python']
                        
                        is_common_topic = any(topic.lower() in prompt.lower() for topic in common_ml_topics)
                        
                        if is_common_topic:
                            # Continue processing with general knowledge approach
                            print(f"No specific course material found for '{prompt}', but recognized as common ML topic.")
                            # We'll add a special flag to the context to indicate this is a general knowledge question
                            context = f"[GENERAL_KNOWLEDGE_QUESTION]\nThe user is asking about a common ML/data science topic.\nQuestion: {prompt}"
                        else:
                            # Inform the user if nothing was found and not a common topic
                            st.warning("Could not find relevant documents in the course material for your query. Please try rephrasing.")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Based on the provided course material context, I cannot answer that question. You may want to rephrase or check the course outline.",
                                "sources": {} # No sources found
                            })
                            st.stop() # Stop further processing for this query if nothing found

                    # 3. Prepare Context and Source Tracking
                    context_parts = []
                    source_to_meta_map = {} # Use this for final display source list

                    for i, doc in enumerate(sources):
                        metadata = doc.metadata
                        source = metadata.get('source', 'Unknown')
                        page_num = metadata.get('page', None)
                        start_index = metadata.get('start_index', None) # From splitter

                        # Format context chunk with metadata
                        context_header = f"Source File: {source}"
                        if page_num:
                            context_header += f" (Page: {page_num})"
                        if start_index is not None:
                            context_header += f" (Approx. Start Index: {start_index})"
                        
                        context_parts.append(f"--- Context Chunk {i+1} ---\n{context_header}\n\n{doc.page_content}\n---")

                        # Store unique source metadata for display
                        source_key = f"{source}_{page_num}" if page_num else source # Unique key per source/page
                        if source_key not in source_to_meta_map:
                            source_to_meta_map[source_key] = metadata

                    context = "\n\n".join(context_parts)
                    
                    # Get conversation history from session state
                    messages = st.session_state.conversation_history.messages
                    if messages:  # Proceed only if there are messages
                        # Limit conversation history to the most recent 10 messages
                        if len(messages) > 10:
                            st.session_state.conversation_history.messages = messages[-10:]
                        # Create a more structured conversation history format
                        conversation_pairs = []
                        for i in range(0, len(messages)-1, 2):  # Process in user-AI pairs
                            if i+1 < len(messages):  # Ensure we have a complete pair
                                user_msg = messages[i].content if messages[i].type == 'human' else 'Unknown question'
                                ai_msg = messages[i+1].content if i+1 < len(messages) and messages[i+1].type == 'ai' else 'Unknown response'
                                conversation_pairs.append(f"User: {user_msg}\nAI: {ai_msg}")
                        
                        # Format the conversation history
                        if conversation_pairs:
                            context_history = "Previous Conversation:\n" + "\n\n".join(conversation_pairs) + "\n\n"
                            # Prepend conversation history to context
                            context = context_history + context
                        
                        print(f"Current conversation history ({len(messages)} messages):")
                        for i, msg in enumerate(messages):
                            print(f"  Message {i+1}: {msg.type} - {msg.content[:50]}...")
                    else:
                        print("Note: No conversation history yet")
                        
                    sources_for_display = source_to_meta_map # Use the unique map for display
                    
                    # Get conversation history from session state messages
                    previous_messages = st.session_state.messages[:-1]  # Exclude current message
                    conversation_history = ""
                    if previous_messages:  # Proceed only if there are previous messages
                        # Limit conversation history to the most recent 10 messages
                        if len(previous_messages) > 10:
                            previous_messages = previous_messages[-10:]
                        
                        # Build a structured conversation history string
                        chat_turns = []
                        for m in previous_messages:
                            role = "Human" if m['role'] == "user" else "Assistant"
                            chat_turns.append(f"{role}: {m['content']}")
                        
                        conversation_history = "\n\n".join(chat_turns)
                        
                        print(f"Current conversation history ({len(previous_messages)} messages):")
                        for i, msg in enumerate(previous_messages):
                            print(f"  Message {i+1}: {msg['role']} - {msg['content'][:50]}...")
                    else:
                        print("Note: No conversation history yet")

                    # 4. Prepare Final Prompt for Gemini
                    # Separate conversation history from content chunks for clarity
                    if conversation_history:
                        conversation_section = f"""
**Previous Conversation:**
{conversation_history}
"""
                    else:
                        conversation_section = ""

                    prompt_template = f"""
You are an AI Tutor for a Machine Learning Bootcamp. Use the provided Context Chunks to answer the user's question.

**Instructions:**
- Answer the user's question using information from the Context Chunks.
- DO NOT use any general knowledge not found in the Context Chunks.
- If you are complimented, feel free to respond in a friendly manner.
- If the answer spans multiple chunks, synthesize the information concisely.
- When the question asks **where/when** a concept appears, mention the Module/Day/Source File/Page (if available) and ensure it's listed in the 'Sources' section later.
- If the question is related to a specific concept (e.g., a formula or method), explain it in a straightforward manner and provide any relevant details from the Context Chunks.
- If the context includes pandas, numpy, or other libraries being asked about, provide an answer based strictly on how they are presented in the course materials.
- IMPORTANT: If the question appears to be a follow-up question, you MUST refer to the Previous Conversation section to understand the context of the conversation.
- Start your response directly with the answer, preceded by `### Answer`.
- Format code examples, commands, or formulas using ```markdown fences``` for clarity.
- Use **bullet points** or **numbered lists** for steps or key items when appropriate.
- If the Context Chunks mention where the concept appears in the course (Module/Day/Source), include this information.
- If the Context Chunks do not contain the answer, clearly state:
  > Based on the provided course material context, I cannot answer that question. You may want to rephrase or check the course content.

{conversation_section}
**Context Chunks:**
{context}

**Current Question:** {prompt}
Answer:"""

                    # 5. Call Gemini API
                    try:
                        generation_model = genai.GenerativeModel("gemini-2.0-flash-exp") # Using stable Gemini model compatible with the current SDK
                        response = generation_model.generate_content(prompt_template)

                        try:
                            # Extract text from the response - different versions of the SDK have different response formats
                            if response:
                                # Try different attributes based on SDK version
                                if hasattr(response, 'text'):
                                    full_response_content = response.text
                                elif hasattr(response, 'parts') and response.parts:
                                    full_response_content = ''.join([part.text for part in response.parts])
                                elif hasattr(response, 'candidates') and response.candidates:
                                    # Handle the older response format
                                    if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                                        full_response_content = ''.join([part.text for part in response.candidates[0].content.parts])
                                    else:
                                        full_response_content = str(response.candidates[0])
                                else:
                                    # Last resort - convert the whole response to string
                                    full_response_content = str(response)
                                
                                message_placeholder.markdown(full_response_content) # Display final answer
                            else:
                                st.error("Failed to get a response from the AI model. The response was empty.")
                        except Exception as format_e:
                            st.error(f"Error formatting response: {format_e}")
                            st.error(f"Raw response: {response}")
                            full_response_content = f"Error formatting response: {format_e}. Raw response: {str(response)[:200]}..."
                            message_placeholder.markdown(full_response_content)

                    except Exception as gen_e:
                        print(f"Error during Gemini API call: {gen_e}")
                        traceback.print_exc()
                        full_response_content = f"Sorry, an error occurred while generating the response: {gen_e}"

            except Exception as e:
                # Catch errors in the main try block (retrieval, context prep)
                st.error(f"An unexpected error occurred: {e}")
                traceback.print_exc()
                full_response_content = f"Sorry, a critical error occurred: {e}"
                message_placeholder.markdown(full_response_content) # Show error in placeholder

            # Add assistant response (or error message) to chat history AFTER processing
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response_content,
                "sources": sources_for_display # Store sources with the message
            })
            
            # Assistant message already added to session state above

elif not gemini_configured:
    st.warning("Gemini API is not configured. Please check your `secrets.toml` file.")
elif faiss_index is None:
    st.warning("FAISS index could not be loaded or built. Chatbot functionality is disabled.")
else:
    st.warning("An unknown configuration error occurred.")
