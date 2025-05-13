# AI Course Tutor - Setup Guide

## Introduction

This guide provides step-by-step instructions for setting up and running the AI Course Tutor application on your local machine. This Streamlit application uses Google Gemini and FAISS to provide answers to questions based on your course materials. Following these steps carefully, especially regarding environment setup, will help ensure a smooth experience.

## Prerequisites

*   **Anaconda or Miniconda:** You need a working installation of Anaconda or Miniconda to manage packages and environments. Download from [Anaconda Distribution](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
*   **Git (Optional):** If you are cloning this project from a Git repository.
*   **Course Materials:** You need the course content files (.pdf, .py, .md, .ipynb, .txt) that the tutor will use.

## Setup Instructions

1.  **Get the Project Code:**
    *   **If using Git:** Clone the repository to your local machine:
        ```bash
        git clone https://github.com/anshimathur/group-4-project-3.git
        cd group-4-project-3
        ```
        *(Note: If you have already cloned or pulled the latest changes for this project repository, you can skip this cloning step and just navigate to the project directory.)*
    *   **If not using Git:** Ensure you have the project files (`app.py`, `requirements.txt`, `create_manifest.py`, etc.) in a dedicated directory on your computer and navigate to that directory in your terminal. Let's refer to this as the `project root`.

2.  **Create a Dedicated Conda Environment (Highly Recommended):**
    *   To avoid package conflicts (like potential OpenMP errors), create a new, clean environment specifically for this project. Open your Anaconda Prompt or terminal.
    *   Choose a Python version (e.g., 3.11 or 3.12 are good choices):
        ```bash
        conda create --name tutor_env python=3.12 -y
        ```
    *   Activate the newly created environment:
        ```bash
        conda activate tutor_env
        ```
    *   You should see `(tutor_env)` at the beginning of your terminal prompt. **Ensure this environment is active for all subsequent steps.**

3.  **Install Dependencies:**
    *   With the `tutor_env` active, install all required Python packages from the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Configure Google API Key:**
    *   The application uses Google Gemini, which requires an API key.
    *   Create a file named `.env` inside your `project root` directory (if it doesn't exist).
    *   Add your Google API key to this file in the following format:
        ```
        GOOGLE_API_KEY=YOUR_ACTUAL_GOOGLE_API_KEY
        ```
    *   Replace `YOUR_ACTUAL_GOOGLE_API_KEY` with your real key.

5.  **Add Course Content:**
    *   Create a directory named `course-content` inside your `project root` directory (if it doesn't exist).
    *   **Crucial Step:** Navigate into the `course-content` directory in your terminal and clone the official course materials repository into it. Then move the transcripts directory into the course-content directory:
        ```bash
        mkdir course-content
        cd course-content/
        git clone https://git.bootcampcontent.com/boot-camp-consortium-west-coast/AI-PT-WEST-NOVEMBER-111824.git .
        rm -rf .git
        rm .gitignore
        cd ..
        mv transcripts/  course-content/
        ```
        *(Note: The `.` at the end of the `git clone` command ensures the content is cloned directly into the `course-content` folder, not into a subfolder named `AI-PT-WEST-NOVEMBER-111824`.)*
    *   The application will recursively scan this folder for `.pdf`, `.py`, `.md`, `.ipynb`, and `.txt` files.

6.  **Generate Content Manifest:**
    *   The application uses a `content_manifest.json` file to track the course materials. You need to generate this file based on the content cloned into the `course-content` directory.
    *   Run the manifest creation script (make sure your `tutor_env` is still active):
        ```bash
        python create_manifest.py
        ```
    *   This script will scan the `course-content` directory and create/update the `content_manifest.json` file in the `project root`.

## Running the Application

1.  **Ensure Environment is Active:** Verify that your terminal prompt starts with `(tutor_env)`.
2.  **Start Streamlit:** Run the following command from the `project root` directory:
    ```bash
    python -m streamlit run Chat.py
    ```
    *   Using `python -m streamlit run` ensures you are using the Python interpreter and packages from your active `tutor_env`.

3.  **First Run - Index Building:**
    *   The first time you run the application after generating the manifest (or if the index file `faiss_index_google_v1` is missing), it will need to build the FAISS vector index.
    *   This involves reading all documents, chunking them, generating embeddings via the Google API, and saving the index.
    *   **This process can take several minutes**, depending on the amount of course content and your internet speed. Please be patient and monitor the terminal output for progress.
    *   Subsequent runs will load the existing index, which is much faster.

4.  **Access the App:** Once the index is built/loaded and Streamlit starts, it will display local and network URLs in the terminal. Open the `Local URL` (usually `http://localhost:8501`) in your web browser to use the AI Course Tutor.

## Troubleshooting

*   **`OMP: Error #15`:** This usually indicates conflicting OpenMP libraries. The best solution is to ensure you created and are using a clean conda environment as described in Step 2.
*   **`KeyError: 'google_api_key'`:** Double-check that you created the `.streamlit/secrets.toml` file correctly (Step 4) and that the key name is exactly `google_api_key`.
*   **`ImportError`:** Make sure you activated the correct conda environment (`conda activate tutor_env`) before running `pip install -r requirements.txt` and before running `python -m streamlit run app.py`.
