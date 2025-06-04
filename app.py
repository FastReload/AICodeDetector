import streamlit as st
import pandas as pd
import config
from evaluate_models import run_full_evaluation
import tempfile
from gemini import set_gemini_api_key, generate_bulk_gemini_codes
from chatgpt import set_chatgpt_api_key, generate_chatgpt_cpp_code

st.set_page_config(page_title="AI Code Detector", layout="centered")
st.title("AI Code Detector")

tabs = st.tabs([
    "\U0001F4D6 Introduction",
    "\U0001F4C1 Upload & Run",
    "\U0001F4CA Compare All Models",
    "\U0001F9EA Try a Code Snippet",
    "\U0001F4D1 Bulk Generate (Gemini)",
    "\U0001F4D1 Bulk Generate (ChatGPT)",
    "\U0001F91D Merge JSONL Files"
])

# ------------------- Tab 0: Introduction -------------------
with tabs[0]:
    st.markdown("""
    ## Welcome to the AI Code Detector App

    This application helps you detect whether a given code snippet is **Human-Written** or **AI-Generated**.  
    It uses advanced machine learning models trained on custom features like **TF-IDF** (via `SCTokenizer`) and **CodeBERT embeddings**.

    ### 🚀 How to Use This App

    #### 1️⃣ Upload Your Data
    - Go to the **Upload & Run** tab.
    - Upload **two `.jsonl` files**:
        - One with human-written code samples (label `0`)
        - One with AI-generated code samples (label `1`)
    - Each line in the JSONL file must follow this format:
    ```json
    {
        "problem": 1,
        "code": "def example(): pass",
        "class": 0,
        "version": 1,
        "model": "Human" or "Gemini",
        "variation": 1
    }
    ```

    #### 2️⃣ Train the Models
    - Click the **Run Models (TF-IDF + CodeBERT)** button.
    - The app trains:
        - **TF-IDF models**: Random Forest, SVM, XGBoost, MLP, and an Ensemble
        - **CodeBERT models**: Same set of models using semantic embeddings

    #### 3️⃣ View and Compare Model Metrics
    - Switch to the **Compare All Models** tab.
    - See accuracy, precision, recall, and F1-score for each model.

    #### 4️⃣ Classify New Code Snippets
    - Go to the **Try a Code Snippet** tab.
    - Paste any code snippet.
    - Choose your tokenizer (TF-IDF or CodeBERT) and model.
    - Get instant predictions on whether the code is human-written or AI-generated!

    #### 5️⃣ Generate AI Code in Bulk
    - Use the **Bulk Generate (Gemini)** tab.
    - Paste prompts and instruction styles using `|` separators.
    - Automatically generate multiple versions of each prompt using Gemini.
    - Download the full dataset for training or analysis.

    #### 6️⃣ Generate AI Code (ChatGPT)
    - Use OpenAI's GPT to generate code in a single-prompt interface.

    ---
    💡 _Need help? Open an issue or contribute on GitHub at [https://github.com/FastReload/AICodeDetector](https://github.com/FastReload/AICodeDetector)._
    """)

# ------------------- Tab 1: Upload & Run -------------------
with tabs[1]:
    uploaded_human = st.file_uploader("Upload Human Code JSONL", type=["jsonl"], key="human")
    uploaded_ai = st.file_uploader("Upload AI Code JSONL", type=["jsonl"], key="ai")

    if uploaded_human and uploaded_ai:
        run_clicked = st.button("Run Models (TF-IDF + CodeBERT)")

        if run_clicked:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_human, tempfile.NamedTemporaryFile(delete=False) as tmp_ai:
                tmp_human.write(uploaded_human.read())
                tmp_ai.write(uploaded_ai.read())
                tmp_human_path = tmp_human.name
                tmp_ai_path = tmp_ai.name

            st.info("Running both TF-IDF and CodeBERT models...")
            progress_bar = st.progress(0)

            def update_progress(current, total):
                progress_bar.progress(current / total)

            metrics_tfidf = run_full_evaluation(
                human_path=tmp_human_path,
                ai_path=tmp_ai_path,
                method="tfidf",
                progress_callback=update_progress
            )

            metrics_codebert = run_full_evaluation(
                human_path=tmp_human_path,
                ai_path=tmp_ai_path,
                method="codebert",
                progress_callback=update_progress
            )

            st.success("### Models Trained! Switch to the next tab to view comparison.")

            st.session_state["models"] = {
                f"TFIDF - {k}": v for k, v in metrics_tfidf.items() if k != "__vectorizer__"
            }
            st.session_state["models"].update({
                f"CodeBERT - {k}": v for k, v in metrics_codebert.items()
            })

            st.session_state["cpp_models"] = {
                k: v["model"] for k, v in metrics_tfidf.items() if "model" in v and k != "Voting Ensemble"
            }
            st.session_state["cpp_vectorizer"] = metrics_tfidf.get("__vectorizer__", {}).get("vectorizer", None)
            st.session_state["cpp_ensemble"] = metrics_tfidf["Voting Ensemble"]["model"]

            st.session_state["bert_models"] = {
                k: v["model"] for k, v in metrics_codebert.items() if "model" in v and k != "Voting Ensemble"
            }
            st.session_state["bert_ensemble"] = metrics_codebert["Voting Ensemble"]["model"]
    else:
        st.warning("Upload both Human and AI code JSONL files to proceed.")

# ------------------- Tab 2: Compare All Models -------------------
with tabs[2]:
    if "models" in st.session_state:
        st.subheader("\U0001F4CA Model Comparison Table")

        tfidf_models = {k: v for k, v in st.session_state["models"].items() if k.startswith("TFIDF")}
        codebert_models = {k: v for k, v in st.session_state["models"].items() if k.startswith("CodeBERT")}

        if tfidf_models:
            st.markdown("### TF-IDF Models")
            df_tfidf = pd.DataFrame(tfidf_models).T.drop(columns=["model", "vectorizer"], errors='ignore')
            st.dataframe(df_tfidf.style.format("{:.4f}"))

        if codebert_models:
            st.markdown("### CodeBERT Models")
            df_codebert = pd.DataFrame(codebert_models).T.drop(columns=["model", "vectorizer"], errors='ignore')
            st.dataframe(df_codebert.style.format("{:.4f}"))

    else:
        st.warning("Upload and run model in the first tab to see results.")

# ------------------- Tab 3: Try a Code Snippet -------------------
with tabs[3]:
    st.subheader("\U0001F9EA Try a Code Snippet")

    if "cpp_models" in st.session_state and "cpp_vectorizer" in st.session_state:
        tokenizer_choice = st.radio("Select Tokenizer", ["CppTokenizer (TF-IDF based)", "CodeBERT (semantic embedding)"])
        tokenizer_type = "TF-IDF" if tokenizer_choice.startswith("Cpp") else "CodeBERT"

        model_names = list(st.session_state["cpp_models"].keys()) + ["Ensemble"]
        selected_model_name = st.selectbox("Select Model", model_names)

        input_code = st.text_area("Paste the code snippet to classify", height=200)

        if st.button("Classify"):
            result = None
            try:
                if tokenizer_type == "TF-IDF":
                    from sctokenizer import CppTokenizer
                    cpp_tokenizer = CppTokenizer()
                    cpp_tokens = cpp_tokenizer.tokenize(input_code)
                    cpp_token_values = ' '.join(token.token_value for token in cpp_tokens)

                    cpp_vectorizer = st.session_state.get("cpp_vectorizer", None)
                    if cpp_vectorizer is None:
                        st.error("TF-IDF vectorizer not found. Please train models with TF-IDF method in Tab 1.")
                    else:
                        features = cpp_vectorizer.transform([cpp_token_values])
                        if selected_model_name == "Ensemble":
                            result = st.session_state["cpp_ensemble"].predict(features)[0]
                        else:
                            result = st.session_state["cpp_models"][selected_model_name].predict(features)[0]

                else:  # CodeBERT
                    from tokenizer_utils import get_codebert_embedding
                    embedding = get_codebert_embedding(input_code).reshape(1, -1)

                    if selected_model_name == "Ensemble":
                        result = st.session_state["bert_ensemble"].predict(embedding)[0]
                    else:
                        result = st.session_state["bert_models"][selected_model_name].predict(embedding)[0]

                if result is not None:
                    st.success(f"Prediction Result using {selected_model_name}")
                    st.markdown("**Classification:** " + ("\U0001F9E0 AI-Generated" if result == 1 else "\U0001F464 Human-Written"))

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please train the models in the first tab before using this feature.")

# ------------------- Tab 4: Bulk Generate (Gemini) -------------------
with tabs[4]:
    st.subheader("🗂 Bulk Generate Code with Gemini")

    st.markdown("""
   
    Use this tab to generate a large dataset of AI-generated code.

    - Enter multiple **prompts** separated by `|`
      *(e.g., `Write a C++ function to add two numbers | Write a program to reverse a string`)*

    - Enter multiple **instruction styles** separated by `|`
      *(e.g., `Write in plain C++ | Beginner-level code | Use basic syntax only`)*

    - The app will generate **10 variations** for **every (prompt × style)** combination.

    - Download the full dataset as `.jsonl` and `.csv`.

    **Example format** in the output:
    ```json
    {
      "problem": 1,
      "code": "#include<iostream>\\nint main() { return 0; }",
      "class": 1,
      "version": 2,
      "model": "Gemini",
      "variation": 3
    }
    ```
    """)

    if "gemini_api_set" not in st.session_state:
        st.session_state["gemini_api_set"] = False

    if not st.session_state["gemini_api_set"]:
        api_key = st.text_input("Enter your Gemini API key:", type="password")
        if st.button("Set Gemini API Key"):
            if not api_key.startswith("AI"):
                st.error("Gemini API key must start with 'AI'.")
            else:
                success = set_gemini_api_key(api_key)
                if success:
                    st.session_state["gemini_api_set"] = True
                    st.success("Gemini API key set successfully!")
                else:
                    st.error("Failed to set Gemini API key.")

    else:
        prompts_input = st.text_area("Enter | separated prompts", height=200, key="gemini_prompts")
        versions_input = st.text_area("Enter | separated instruction styles", height=150, key="gemini_versions")

        if st.button("Run Bulk Generation"):
            if not prompts_input.strip() or not versions_input.strip():
                st.error("Both fields are required.")
            else:
                progress_bar = st.progress(0.0)
                with st.spinner("Generating C++ code with Gemini..."):
                    jsonl_path, csv_path = generate_bulk_gemini_codes(
                        prompts_input, versions_input,
                        progress_callback=lambda p: progress_bar.progress(p)
                    )
                st.success("Generation complete!")
                with open(jsonl_path, "rb") as f:
                    st.download_button("Download JSONL", f, file_name="gemini_generated.jsonl")
                with open(csv_path, "rb") as f:
                    st.download_button("Download CSV", f, file_name="gemini_generated.csv")

# ------------------- Tab 5: Bulk Generate (ChatGPT) -------------------
with tabs[5]:
    st.subheader("🗂 Bulk Generate Code with ChatGPT")

    st.markdown("""
    Use this tab to generate a large dataset of AI-generated code.

    - Enter multiple **prompts** separated by `|`
      *(e.g., `Write a C++ function to add two numbers | Write a program to reverse a string`)*

    - Enter multiple **instruction styles** separated by `|`
      *(e.g., `Write in plain C++ | Beginner-level code | Use basic syntax only`)*

    - The app will generate **10 variations** for **every (prompt × style)** combination.

    - Download the full dataset as `.jsonl` and `.csv`.

    **Example format** in the output:
    ```json
    {
      "problem": 1,
      "code": "#include<iostream>\\nint main() { return 0; }",
      "class": 1,
      "version": 2,
      "model": "GPT-4o",
      "variation": 3
    }
    ```
    """)

    if "chatgpt_api_set" not in st.session_state:
        st.session_state["chatgpt_api_set"] = False

    if not st.session_state["chatgpt_api_set"]:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if st.button("Set ChatGPT API Key"):
            if not api_key.startswith("sk-"):
                st.error("OpenAI API key must start with 'sk-'.")
            else:
                success = set_chatgpt_api_key(api_key)
                if success:
                    st.session_state["chatgpt_api_set"] = True
                    st.success("API key set successfully!")
                else:
                    st.error("Failed to set OpenAI API key.")
    else:
        prompts_input = st.text_area("Enter | separated prompts", height=200, key="chatgpt_prompts")
        versions_input = st.text_area("Enter | separated instruction styles", height=150, key="chatgpt_versions")
        
        if st.button("Run ChatGPT Bulk Generation"):
            if not prompts_input.strip() or not versions_input.strip():
                st.error("Please enter both prompts and instruction styles.")
            else:
                from chatgpt import generate_bulk_chatgpt_codes
                progress_bar = st.progress(0.0)
                with st.spinner("Generating C++ code with ChatGPT..."):
                    jsonl_path, csv_path = generate_bulk_chatgpt_codes(
                        prompts_input, versions_input,
                        progress_callback=lambda p: progress_bar.progress(p)
                    )
                st.success("Bulk code generation complete.")
                with open(jsonl_path, "rb") as f:
                    st.download_button("Download JSONL", f, file_name="gpt4_generated.jsonl")
                with open(csv_path, "rb") as f:
                    st.download_button("Download CSV", f, file_name="gpt4_generated.csv")

# ------------------- Tab 6: Merge JSONL Files -------------------
with tabs[6]:
    st.subheader("🔀 Merge Multiple JSONL Files")

    st.markdown("""
    Use this tab to combine multiple `.jsonl` files into one.

    - Upload two or more `.jsonl` files
    - Enter a desired name for the output file
    - Click **Merge Files** to generate and download the result

    Each uploaded file must follow the format:
    ```json
    {
      "problem": 1,
      "code": "#include<iostream>\\nint main() { return 0; }",
      "class": 1,
      "version": 1,
      "model": "Gemini",
      "variation": 1
    }
    ```
    """)

    uploaded_files = st.file_uploader(
        "Upload multiple JSONL files to merge", type=["jsonl"], accept_multiple_files=True
    )

    output_name = st.text_input("Enter output filename (without extension):", "merged_output")

    if uploaded_files and st.button("Merge Files"):
        merged_path = f"{output_name.strip()}.jsonl"
        try:
            with open(merged_path, "w", encoding="utf-8") as outfile:
                for file in uploaded_files:
                    contents = file.read().decode("utf-8")
                    for line in contents.splitlines():
                        outfile.write(line + "\n")

            st.success(f"Merged {len(uploaded_files)} files into `{merged_path}`")

            with open(merged_path, "rb") as f:
                st.download_button("Download Merged File", f, file_name=merged_path)
        except Exception as e:
            st.error(f"Error during merge: {e}")
