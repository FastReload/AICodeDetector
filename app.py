import streamlit as st
import pandas as pd
import config
from evaluate_models import run_full_evaluation
import tempfile

st.set_page_config(page_title="AI Code Detector", layout="centered")
st.title("AI Code Detector")

tabs = st.tabs(["\U0001F4C1 Upload & Run", "\U0001F4CA Compare All Models", "\U0001F9EA Try a Code Snippet"])

metrics_dict = {}

with tabs[0]:
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

            # Store in session
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

with tabs[1]:
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

with tabs[2]:
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