import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import config
from evaluate_models import run_full_evaluation
import tempfile
from datetime import datetime
from gemini import set_gemini_api_key, generate_bulk_gemini_codes
from chatgpt import set_chatgpt_api_key, generate_chatgpt_cpp_code, generate_bulk_chatgpt_codes
from gradient_explainer import create_gradient_explainer

st.set_page_config(
    page_title="AI Code Detector | Advanced ML Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="AI"
)

# Spotify-inspired minimalist CSS (orange accent)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 900;
        color: #f97316;
        letter-spacing: -0.04em;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #b3b3b3;
        font-weight: 400;
        margin-bottom: 3rem;
        letter-spacing: -0.01em;
    }
    
    /* Tab styling - Spotify-like */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
        border-bottom: 1px solid #282828;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        color: #b3b3b3;
        border: none;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0 24px;
        transition: color 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #f97316;
        border-bottom: 2px solid #f97316;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
    }
    
    /* Card styling - dark, minimal */
    .content-card {
        background: #181818;
        padding: 24px;
        border-radius: 8px;
        margin: 16px 0;
        transition: background 0.3s;
    }
    
    .content-card:hover {
        background: #282828;
    }
    
    /* Button styling - orange accent */
    .stButton>button {
        background: #f97316;
        color: #ffffff;
        border: none;
        border-radius: 500px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: #fb923c;
        transform: scale(1.04);
    }
    
    /* Clean section headers */
    h3 {
        font-weight: 700;
        font-size: 1.5rem;
        color: #ffffff;
        margin-top: 2rem;
        letter-spacing: -0.02em;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: #181818;
        border-radius: 8px;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 3rem;
        max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">AI Code Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect AI-generated code with machine learning</p>', unsafe_allow_html=True)

tabs = st.tabs([
    "Introduction",
    "Upload & Run",
    "Compare All Models",
    "Try a Code Snippet",
    "Bulk Generate (Gemini)",
    "Bulk Generate (ChatGPT)",
    "Merge JSONL Files"
])

# ------------------- Tab 0: Introduction -------------------
with tabs[0]:
    st.markdown(
        """
<div class='content-card'>
    <h2 style='color: #f97316; margin-top: 0; font-weight: 700;'>Welcome</h2>
    <p style='font-size: 1.1rem; line-height: 1.8; color: #b3b3b3;'>
        Train machine learning models to distinguish between human-written and AI-generated code.
        Uses TF-IDF and UniXcoder embeddings for accurate detection.
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Getting Started")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
<div class='content-card'>
    <h4 style='color: #ffffff; font-weight: 600;'>1. Upload Data</h4>
    <p style='color: #b3b3b3;'>Upload two JSONL files with human and AI code samples</p>
    <br/>
    <h4 style='color: #ffffff; font-weight: 600;'>2. Train Models</h4>
    <p style='color: #b3b3b3;'>Train 10 different ML models with one click</p>
    <br/>
    <h4 style='color: #ffffff; font-weight: 600;'>3. Analyze</h4>
    <p style='color: #b3b3b3;'>Compare performance and test new code snippets</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div class='content-card'>
    <h4 style='color: #ffffff; font-weight: 600;'>Key Features</h4>
    <p style='color: #b3b3b3; margin-bottom: 1rem;'>
        <b style='color: #f97316;'>Explainability:</b> See which code tokens trigger detection
    </p>
    <p style='color: #b3b3b3; margin-bottom: 1rem;'>
        <b style='color: #f97316;'>Mixed Authorship:</b> Line-by-line analysis detects copy-pasted AI code
    </p>
    <p style='color: #b3b3b3; margin-bottom: 1rem;'>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Data Format")
    st.code(
        '''{
  "problem": 1,
  "code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
  "class": 0,
  "version": 1,
  "model": "Human"
}''',
        language="json",
    )

# ------------------- Tab 1: Upload & Run -------------------
with tabs[1]:
    st.markdown("### Upload Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
<div class='content-card'>
    <h4 style='color: #ffffff; font-weight: 600;'>Human Code</h4>
    <p style='color: #b3b3b3; font-size: 0.9rem;'>JSONL file with human-written samples</p>
</div>
        """, unsafe_allow_html=True)
        uploaded_human = st.file_uploader("Choose file", type=["jsonl"], key="human", label_visibility="collapsed")
    
    with col2:
        st.markdown("""
<div class='content-card'>
    <h4 style='color: #ffffff; font-weight: 600;'>AI Code</h4>
    <p style='color: #b3b3b3; font-size: 0.9rem;'>JSONL file with AI-generated samples</p>
</div>
        """, unsafe_allow_html=True)
        uploaded_ai = st.file_uploader("Choose file", type=["jsonl"], key="ai", label_visibility="collapsed")

    if uploaded_human and uploaded_ai:
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Dataset Analysis Section
        with st.expander("📊 **Dataset Analysis**", expanded=True):
            try:
                # Load both datasets
                human_data = [json.loads(line) for line in uploaded_human.getvalue().decode('utf-8').splitlines()]
                ai_data = [json.loads(line) for line in uploaded_ai.getvalue().decode('utf-8').splitlines()]
                
                # Reset file pointers for later use
                uploaded_human.seek(0)
                uploaded_ai.seek(0)
                
                # Calculate statistics
                human_count = len(human_data)
                ai_count = len(ai_data)
                total_count = human_count + ai_count
                
                # Code length statistics
                human_lengths = [len(item.get('code', '').splitlines()) for item in human_data]
                ai_lengths = [len(item.get('code', '').splitlines()) for item in ai_data]
                
                # Token count statistics (approximate - split by whitespace)
                human_tokens = [len(item.get('code', '').split()) for item in human_data]
                ai_tokens = [len(item.get('code', '').split()) for item in ai_data]
                
                # Display metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Total Samples", total_count)
                with metric_cols[1]:
                    st.metric("Human Code", human_count)
                with metric_cols[2]:
                    st.metric("AI Code", ai_count)
                with metric_cols[3]:
                    balance_ratio = min(human_count, ai_count) / max(human_count, ai_count) * 100
                    st.metric("Balance", f"{balance_ratio:.1f}%")
                
                st.markdown("<br/>", unsafe_allow_html=True)
                
                # Statistics comparison
                stat_cols = st.columns(2)
                
                with stat_cols[0]:
                    st.markdown("**Human Code Statistics**")
                    st.write(f"• Avg Lines: {sum(human_lengths)/len(human_lengths):.1f}")
                    st.write(f"• Avg Tokens: {sum(human_tokens)/len(human_tokens):.1f}")
                    st.write(f"• Min/Max Lines: {min(human_lengths)} / {max(human_lengths)}")
                
                with stat_cols[1]:
                    st.markdown("**AI Code Statistics**")
                    st.write(f"• Avg Lines: {sum(ai_lengths)/len(ai_lengths):.1f}")
                    st.write(f"• Avg Tokens: {sum(ai_tokens)/len(ai_tokens):.1f}")
                    st.write(f"• Min/Max Lines: {min(ai_lengths)} / {max(ai_lengths)}")
                
                # Visualizations
                st.markdown("<br/>", unsafe_allow_html=True)
                
                chart_cols = st.columns(2)
                
                with chart_cols[0]:
                    # Distribution pie chart
                    import pandas as pd
                    df_dist = pd.DataFrame({
                        'Category': ['Human Code', 'AI Code'],
                        'Count': [human_count, ai_count]
                    })
                    st.markdown("**Dataset Distribution**")
                    st.bar_chart(df_dist.set_index('Category'))
                
                with chart_cols[1]:
                    # Length comparison
                    df_lengths = pd.DataFrame({
                        'Human (avg lines)': [sum(human_lengths)/len(human_lengths)],
                        'AI (avg lines)': [sum(ai_lengths)/len(ai_lengths)]
                    })
                    st.markdown("**Average Code Length**")
                    st.bar_chart(df_lengths)
                
            except Exception as e:
                st.error(f"Error analyzing dataset: {str(e)}")
        
        st.markdown("<br/>", unsafe_allow_html=True)
        run_clicked = st.button("Train Models", use_container_width=True)

        if run_clicked:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_human, tempfile.NamedTemporaryFile(delete=False) as tmp_ai:
                tmp_human.write(uploaded_human.read())
                tmp_ai.write(uploaded_ai.read())
                tmp_human_path = tmp_human.name
                tmp_ai_path = tmp_ai.name

            st.info("Running both TF-IDF and UniXcoder models...")
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

            st.markdown("""
<div class='content-card' style='background: #f97316; color: #ffffff;'>
    <h3 style='color: #ffffff; margin: 0; font-weight: 700;'>Training Complete</h3>
    <p style='margin: 0.5rem 0 0 0; color: #ffffff;'>View results in the <b>Compare All Models</b> tab</p>
</div>
            """, unsafe_allow_html=True)

            st.session_state["models"] = {
                f"TFIDF - {k}": v for k, v in metrics_tfidf.items() if k != "__vectorizer__"
            }
            st.session_state["models"].update({
                f"UniXcoder - {k}": v for k, v in metrics_codebert.items()
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
        st.markdown("### Model Performance")
        st.markdown("<p style='color: #b3b3b3;'>Accuracy, precision, recall, and F1-score for all trained models</p>", unsafe_allow_html=True)
        
        tfidf_models = {k: v for k, v in st.session_state["models"].items() if k.startswith("TFIDF")}
        unixcoder_models = {k: v for k, v in st.session_state["models"].items() if k.startswith("UniXcoder")}

        if tfidf_models:
            st.markdown("<h4 style='color: #f97316; font-weight: 600; margin-top: 2rem;'>TF-IDF Models</h4>", unsafe_allow_html=True)
            df_tfidf = pd.DataFrame(tfidf_models).T.drop(columns=["model", "vectorizer"], errors='ignore')
            st.dataframe(df_tfidf.style.format("{:.4f}"), use_container_width=True, height=250)

        if unixcoder_models:
            st.markdown("<h4 style='color: #f97316; font-weight: 600; margin-top: 2rem;'>UniXcoder Models</h4>", unsafe_allow_html=True)
            df_unixcoder = pd.DataFrame(unixcoder_models).T
            cols_to_drop = [c for c in ["model", "vectorizer"] if c in df_unixcoder.columns]
            if cols_to_drop:
                df_unixcoder = df_unixcoder.drop(columns=cols_to_drop)
            st.dataframe(df_unixcoder.style.format("{:.4f}"), use_container_width=True, height=250)

    else:
        st.warning("Upload and run model in the first tab to see results.")

# ------------------- Tab 3: Try a Code Snippet -------------------
with tabs[3]:
    st.subheader("Try a Code Snippet")

    if "cpp_models" in st.session_state and "cpp_vectorizer" in st.session_state:
        tokenizer_choice = st.radio("Select Tokenizer", ["CppTokenizer (TF-IDF based)", "UniXcoder (semantic embedding)"])
        tokenizer_type = "TF-IDF" if tokenizer_choice.startswith("Cpp") else "UniXcoder"

        model_names = list(st.session_state["cpp_models"].keys()) + ["Ensemble"]
        selected_model_name = st.selectbox("Select Model", model_names)

        input_code = st.text_area("Paste the code snippet to classify", height=200)

        if st.button("Classify"):
            result = None
            probability = None
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
                            model = st.session_state["cpp_ensemble"]
                            result = model.predict(features)[0]
                            probability = model.predict_proba(features)[0]
                        else:
                            model = st.session_state["cpp_models"][selected_model_name]
                            result = model.predict(features)[0]
                            probability = model.predict_proba(features)[0]

                else:  # CodeBERT
                    from tokenizer_utils import get_codebert_embedding
                    embedding = get_codebert_embedding(input_code).reshape(1, -1)

                    if selected_model_name == "Ensemble":
                        model = st.session_state["bert_ensemble"]
                        result = model.predict(embedding)[0]
                        probability = model.predict_proba(embedding)[0]
                    else:
                        model = st.session_state["bert_models"][selected_model_name]
                        result = model.predict(embedding)[0]
                        probability = model.predict_proba(embedding)[0]

                if result is not None:
                    # Region-level (mixed authorship) analysis
                    with st.expander("Region Analysis", expanded=True):
                        st.markdown("### Region-level Analysis · Mixed Authorship Detection")
                        st.markdown(
                            """**What is this?** Similar to GPTZero's sentence-level detection, this analyzes your code line-by-line using a sliding window to spot mixed authorship, including copy-pasted AI code mixed with human code."""
                        )
                        
                        try:
                            from region_authorship import (
                                analyze_mixed_authorship, 
                                render_line_heatmap_html,
                                format_suspicious_regions_text,
                                calculate_optimal_window_size,
                                generate_risk_assessment
                            )
                            
                            # Auto-calculate optimal window size
                            num_lines = len(input_code.splitlines())
                            auto_window = calculate_optimal_window_size(num_lines)
                            auto_stride = max(2, auto_window // 3)
                            
                            st.info(f"Auto-detected optimal window: **{auto_window} lines** (stride: {auto_stride}) based on {num_lines} lines of code")
                            st.caption("📊 **Baseline Normalization Active**: Probabilities are adjusted relative to boilerplate (includes, using namespace, etc.) to remove systematic bias and focus on actual code patterns.")
                            
                            # Always use auto-calculated window/stride (no manual toggle)
                            window_size = auto_window
                            stride_size = auto_stride
                            
                            # Create predictor based on tokenizer type
                            if tokenizer_type == "TF-IDF":
                                from code_explainer import CodeExplainer
                                explainer = CodeExplainer(model, cpp_vectorizer)
                                predict_func = explainer.predict_ai_probability
                            else:  # UniXcoder
                                from gradient_explainer import create_gradient_explainer
                                explainer = create_gradient_explainer(model)
                                predict_func = explainer.predict_ai_probability
                            
                            with st.spinner("Analyzing code regions..."):
                                region_result = analyze_mixed_authorship(
                                    code=input_code,
                                    predict_ai_proba=predict_func,
                                    window_lines=window_size,
                                    stride_lines=stride_size,
                                    top_k=10,
                                    region_threshold=0.7
                                )
                            
                            # Generate risk assessment
                            risk_level, risk_emoji, risk_message = generate_risk_assessment(
                                region_result.per_line_ai_prob,
                                region_result.avg_ai_prob
                            )
                            
                            # Display risk assessment prominently
                            if risk_level == "high":
                                st.error(f"{risk_emoji} {risk_message}")
                            elif risk_level in ["medium-high", "medium"]:
                                st.warning(f"{risk_emoji} {risk_message}")
                            elif risk_level == "low-medium":
                                st.info(f"{risk_emoji} {risk_message}")
                            else:
                                st.success(f"{risk_emoji} {risk_message}")
                            
                            # Summary metrics
                            st.markdown("### Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Max AI Probability", f"{region_result.max_ai_prob:.1%}")
                            with col2:
                                suspicious_count = len(region_result.suspicious_regions)
                                st.metric("Suspicious Regions", str(suspicious_count))
                            
                            # Heatmap visualization
                            st.markdown("### Line-by-Line Heatmap")
                            st.markdown("**Color coding 🟢🟡🔴:** Uncolored/Gray = low risk (likely human), Yellow = medium (needs discussion), Red = high AI probability")
                            heatmap_html = render_line_heatmap_html(
                                input_code,
                                region_result.per_line_ai_prob,
                                show_line_numbers=True,
                                threshold=0.7
                            )
                            components.html(heatmap_html, height=min(400, max(200, len(input_code.splitlines()) * 25)), scrolling=True)
                            
                            # Suspicious regions
                            if region_result.suspicious_regions:
                                pass  # Heatmap already visualizes flagged regions
                            else:
                                # If no red regions, tailor message based on overall risk
                                if risk_level in ["medium", "medium-high"]:
                                    st.warning("Moderate AI signatures detected (yellow lines). **Discussion recommended.**")
                                elif risk_level == "low-medium":
                                    st.info("Some AI-like patterns detected. Consider a brief explanation from the student.")
                                else:
                                    st.success("No highly suspicious regions detected. Code appears uniformly consistent.")
                            
                            # Most suspicious lines
                            if region_result.most_suspicious_lines:
                                st.markdown("### Most Suspicious Individual Lines")
                                suspicious_df = pd.DataFrame(
                                    [(f"Line {ln}", f"{p:.1%}", txt[:80] + "..." if len(txt) > 80 else txt) 
                                     for (ln, p, txt) in region_result.most_suspicious_lines[:10]],
                                    columns=['Line', 'AI Probability', 'Code Preview']
                                )
                                st.dataframe(suspicious_df, use_container_width=True)
                            
                            st.info("""
**Interpretation Guide**
- 🟢 <30%: Likely human-written (boilerplate and standard code patterns)
- 🟡 30-70%: Moderate AI signature — needs discussion/context from student
- 🔴 >70%: Strong AI signature — high risk, likely AI-generated
- 🟢🟡🔴 Multiple red regions: Probable mixed authorship (AI copy-pasted with human code)

**Note on Boilerplate:** Include statements, using namespace, and standard imports are weighted at 75% of their detected probability to focus analysis on actual code logic.

**Action Items**
- 🔴 **Red regions:** Request detailed code walkthrough and explanation
- 🟡 **Yellow regions:** Ask for reasoning/thought process behind implementation
- 🟢 **Green regions:** Likely acceptable; spot-check edge cases only
                            """)
                            
                        except Exception as e:
                            st.error(f"Error during region analysis: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please train the models in the first tab before using this feature.")

# ------------------- Tab 4: Bulk Generate (Gemini) -------------------
with tabs[4]:
    st.subheader("Bulk Generate Code with Gemini")

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
            if not api_key:
                st.error("Please enter a valid API key.")
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
    st.subheader("Bulk Generate Code with ChatGPT")

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
        api_key = st.text_input("Enter your OpenAI API key:", type="password", key="chatgpt_api_input")
        if st.button("Set ChatGPT API Key"):
            if not api_key:
                st.error("Please enter a valid API key.")
            else:
                with st.spinner("Validating API key..."):
                    success = set_chatgpt_api_key(api_key)
                if success:
                    st.session_state["chatgpt_api_set"] = True
                    st.success("ChatGPT API key set successfully!")
                else:
                    st.error("Failed to set ChatGPT API key. Please check your API key and try again.")
    else:
        prompts_input = st.text_area("Enter | separated prompts", height=200, key="chatgpt_prompts")
        versions_input = st.text_area("Enter | separated instruction styles", height=150, key="chatgpt_versions")
        
        if st.button("Run Bulk Generation", key="chatgpt_run"):
            if not prompts_input.strip() or not versions_input.strip():
                st.error("Both fields are required.")
            else:
                progress_bar = st.progress(0.0)
                with st.spinner("Generating C++ code with ChatGPT..."):
                    jsonl_path, csv_path = generate_bulk_chatgpt_codes(
                        prompts_input, versions_input,
                        progress_callback=lambda p: progress_bar.progress(p)
                    )
                st.success("Generation complete!")
                with open(jsonl_path, "rb") as f:
                    st.download_button("Download JSONL", f, file_name="gpt5_generated.jsonl")
                with open(csv_path, "rb") as f:
                    st.download_button("Download CSV", f, file_name="gpt5_generated.csv")

# ------------------- Tab 6: Merge JSONL Files -------------------
with tabs[6]:
    st.subheader("Merge Multiple JSONL Files")

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