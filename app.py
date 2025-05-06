import streamlit as st
import json
from pathlib import Path
import pandas as pd
import nbformat # To read notebooks
import re
import google.generativeai as genai # Add Gemini import
from pypdf import PdfReader # Added for PDF reading
import traceback # Import for printing traceback
import time # To check manifest modification time

# Langchain & Embedding specific imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # Updated import
from langchain_community.embeddings import SentenceTransformerEmbeddings # Updated import
from langchain.docstore.document import Document # Correct import
from collections import Counter


# --- Configuration ---
st.set_page_config(layout="wide", page_title="AI Course Tutor")

# --- Constants ---
MANIFEST_PATH = Path(__file__).parent / "content_manifest.json"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index_v2" # Use a new folder for the revised index
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Common choice
COURSE_CONTENT_ROOT = Path(__file__).parent / "course-content" # Define root for relative paths

# --- Gemini Configuration ---
gemini_configured = False
try:
    GOOGLE_API_KEY = st.secrets["google_api_key"]
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Use a capable model
    gemini_configured = True
    print("Gemini configured successfully.")
except KeyError:
    st.error("Google API Key not found in Streamlit secrets (secrets.toml). Please add `google_api_key = 'YOUR_API_KEY'`")
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
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: # Only add pages with extracted text
                    pages_content.append((i + 1, page_text)) # Store 1-based page number
            except Exception as page_e:
                print(f"Warning: Error extracting text from page {i+1} in PDF {file_path}: {page_e}")
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
# Note: PDF reader now returns a list, handled in build_or_load_index
READERS = {
    '.py': read_py_file,
    '.md': read_md_file,
    '.ipynb': read_ipynb_file,
    '.pdf': read_pdf_file,
    '.txt': read_txt_file
}

# --- Build or Load FAISS Index ---

# @st.cache_resource # Re-enable caching once stable
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
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750, # Slightly smaller chunks
            chunk_overlap=100, # Smaller overlap
            length_function=len,
            add_start_index=True, # Helps locate chunk origin within source
        )

        for index, row in df.iterrows():
            processed_count += 1
            relative_path = row.get('relative_path')
            if not relative_path:
                print(f"Warning: Skipping row {index} due to missing 'relative_path'.")
                continue

            file_ext = Path(relative_path).suffix.lower()
            reader = READERS.get(file_ext)
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
                                docs_split = text_splitter.create_documents([page_content], metadatas=[page_metadata])
                                all_docs_for_faiss.extend(docs_split)
                    elif isinstance(content_or_pages, str) and content_or_pages:
                        # Handle non-PDF content
                        docs_split = text_splitter.create_documents([content_or_pages], metadatas=[base_metadata])
                        all_docs_for_faiss.extend(docs_split)
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

        file_type_counts = Counter(doc.metadata['file_type'] for doc in all_docs_for_faiss)
        print("🔍 Indexed document chunks by type:", file_type_counts)

        try:
            print(f"Creating FAISS index from {len(all_docs_for_faiss)} document chunks...")
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            faiss_index = FAISS.from_documents(all_docs_for_faiss, embeddings)

            FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
            faiss_index.save_local(FAISS_INDEX_PATH.as_posix())
            print(f"FAISS index successfully built and saved to {FAISS_INDEX_PATH}")
            return faiss_index
        except Exception as e:
            st.error(f"Fatal Error creating or saving FAISS index: {e}")
            traceback.print_exc()
            return None

    return None # Should not be reached

# --- Load Content Manifest ---
# @st.cache_data # Cache the manifest loading
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
st.title("AI Course Tutor Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist for assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Sources Used", expanded=False):
                for source_path, meta in message["sources"].items():
                    page_info = f", Page {meta.get('page')}" if 'page' in meta else ""
                    st.write(f"- **{source_path}**{page_info}")
                    st.caption(f"  (Module: {meta.get('module', 'N/A')}, Day: {meta.get('day', 'N/A')}, Type: {meta.get('file_type', 'N/A')})")
                    if meta.get('slideshow') and meta['slideshow'] != 'N/A':
                         st.caption(f"  Related Slideshow: `{meta['slideshow']}`")


# --- Main Chat Logic ---
if faiss_index is not None and gemini_configured:
    if prompt := st.chat_input("Ask something about the course material..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare for assistant response
        full_response_content = "Error: Response generation failed."
        sources_for_display = {}
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Placeholder for streaming/final answer
            try:
                with st.spinner("Searching course material and generating answer..."):
                    # 1. Retrieve relevant documents
                    k_results = 7 # Retrieve slightly more
                    score_threshold = 1.2 # Adjusted threshold for all-MiniLM-L6-v2 L2 distance

                    # print(f"DEBUG: Performing similarity search for: '{prompt}' with k={k_results}")
                    search_results_with_scores = faiss_index.similarity_search_with_score(prompt, k=k_results)
                    print("🔎 Raw Top-K Search Results (pre-filter):")
                    for doc, score in search_results_with_scores:
                        print(f"  score={score:.2f}  file={doc.metadata['source']}  type={doc.metadata['file_type']}")
                    
                    # print(f"DEBUG: Initial search results count: {len(search_results_with_scores)}")
                    # print(f"DEBUG: Scores: {[score for _, score in search_results_with_scores]}")

                    # 2. Filter results
                    filtered_results = [(doc, score) for doc, score in search_results_with_scores if score < score_threshold]
                    # print(f"DEBUG: Filtered results count (threshold < {score_threshold}): {len(filtered_results)}")

                    if not filtered_results:
                        # If filtering removed everything, maybe take the single best result if it exists?
                        # Or just inform the user. Let's inform for now.
                        if search_results_with_scores: # Check if there were any results initially
                             st.warning(f"Could not find highly relevant documents (closest score: {search_results_with_scores[0][1]:.2f}, threshold: {score_threshold}). The answer might be less accurate or unavailable. Consider rephrasing.")
                             # Fallback: use the single best result despite score
                             # search_results_docs = [search_results_with_scores[0][0]]
                             # Let's just use an empty context if nothing meets threshold
                             search_results_docs = []
                             
                        else:
                            st.error("Could not find any relevant documents in the course material for your query.")
                            full_response_content = "I couldn't find any relevant documents in the course material matching your question."
                            # Set response directly and skip Gemini call
                            message_placeholder.markdown(full_response_content)
                             # Add to history below, outside the try/except for Gemini
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response_content,
                                "sources": {} # No sources found
                            })
                            st.stop() # Stop further processing for this query if nothing found
                    else:
                         search_results_docs = [doc for doc, score in filtered_results]


                    # 3. Prepare Context and Source Tracking
                    context_parts = []
                    source_to_meta_map = {} # Use this for final display source list

                    for i, doc in enumerate(search_results_docs):
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
                    sources_for_display = source_to_meta_map # Use the unique map for display

                    # 4. Prepare Final Prompt for Gemini (No History)
                    prompt_template = f"""
You are an AI Tutor for a Machine Learning Bootcamp. Use ONLY the provided Context Chunks.     
  - `### Answer` heading for your explanation.  
  - Use **bullet points** or **numbered lists** for multi-step items.  
  - Wrap any code or formulas in fenced blocks (```…```).      
  - Do **not** repeat full file names inline.  

**Content Rules**  
1. Base your answer **only** on Context Chunks below.  
2. If the answer is in a chunk, synthesize it; don’t add outside info.  
3. If the question asks **where/when** a concept appears, include its Module/Day/Source/File/Page in your Sources list.  
4. If no chunk covers the question, reply exactly:  
   > “Based on the provided course material context, I cannot answer that question. You may want to rephrase or check the course outline.”  

---

**Context Chunks:**  
{context}

**User Question:** {prompt}

### Answer
"""

                    # 5. Call Gemini API
                    # print("DEBUG: Calling Gemini API...") # Optional debug
                    # print("--- PROMPT START ---")
                    # print(prompt_template)
                    # print("--- PROMPT END ---")

                    try:
                        response = gemini_model.generate_content(prompt_template)
                        # print("DEBUG: Gemini Response Received.") # Optional debug
                        gemini_answer = response.text
                    except ValueError as ve:
                         # Handle potential value errors during text extraction (e.g., blocked content)
                         print(f"Warning: ValueError accessing Gemini response text: {ve}")
                         gemini_answer = "Assistant Error: The response from the AI model could not be processed. It might have been blocked due to safety settings or contained no text."
                         # Consider logging response.parts or response.prompt_feedback here if needed
                         # print(f"DEBUG: Gemini Response Parts: {response.parts}")
                         # print(f"DEBUG: Gemini Prompt Feedback: {response.prompt_feedback}")
                    except Exception as gen_e:
                         print(f"Error during Gemini API call: {gen_e}")
                         traceback.print_exc()
                         gemini_answer = f"Sorry, an error occurred while generating the response: {gen_e}"

                    full_response_content = gemini_answer
                    message_placeholder.markdown(full_response_content) # Display final answer

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
            # Update the sources expander for the *newly added* message immediately
            # (This requires rerunning the loop that displays messages, which Streamlit handles automatically)
            # st.rerun() # Force rerun might be too disruptive, usually updates automatically


elif not gemini_configured:
    st.warning("Gemini API is not configured. Please check your `secrets.toml` file.")
elif faiss_index is None:
     st.warning("FAISS index could not be loaded or built. Chatbot functionality is disabled.")
else:
     st.warning("An unknown configuration error occurred.")
