# AI Code Detector

A sophisticated Streamlit web application designed to distinguish between human-written and AI-generated C/C++ code. The system leverages TF-IDF and UniXcoder embeddings combined with advanced machine learning models to perform highly accurate classification and mixed-authorship detection.

## Features

### Core Detection
* **Upload and Train:** Upload JSONL files containing human-written and AI-generated code samples. The system trains multiple ML models using both TF-IDF vectorization and UniXcoder embeddings.
* **Model Comparison:** Compare performance metrics across Random Forest, SVM, XGBoost, MLP, and Voting Ensemble models through an interactive interface.
* **Interactive Classification:** Real-time code snippet classification with AI probability scores.

### Advanced Analysis
* **Mixed-Authorship Detection:** Line-by-line analysis using sliding window techniques to identify potential copy-pasted AI code mixed with human code.
* **Baseline Normalization:** Intelligent filtering that reduces false positives from boilerplate code (includes, namespace declarations, etc.).
* **Region Heatmap Visualization:** Visual representation of suspicious code regions with color-coded AI probability scores.
* **Risk Assessment:** Automated flagging of high-risk code sections that likely contain AI-generated content.

### AI Code Generation
* **Gemini Integration:** Bulk generate C++ code datasets using Google's Gemini API with customizable prompts and instruction styles.
* **ChatGPT Integration:** Generate code variations using OpenAI's GPT-5.1 API for dataset creation and testing.
* **JSONL Utilities:** Merge multiple JSONL files for dataset management.

## Setup

### Clone Repository

```bash
git clone https://github.com/FastReload/AICodeDetector
cd AICodeDetector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

This command launches a browser interface with seven tabs:

* **Introduction:** Overview of the detection system and key features.
* **Upload & Run:** Upload JSONL files containing human and AI code samples to train models.
* **Compare All Models:** View detailed performance metrics for all trained models.
* **Try a Code Snippet:** Classify individual code snippets with AI probability scores and mixed-authorship analysis.
* **Bulk Generate (Gemini):** Generate large C++ code datasets using Google Gemini API.
* **Bulk Generate (ChatGPT):** Generate code variations using OpenAI GPT-5.1 API.
* **Merge JSONL Files:** Combine multiple JSONL datasets into a single file.

## Data Format

Input files must be in `.jsonl` format. Each line should follow the structure:

```json
{"code": "def example_function(): pass"}
```

* Human-written samples must have label `0`
* AI-generated samples must have label `1`

### Example

```json
{"problem": 2, "code": "#include <iostream>\n#include <iomanip>\n\nusing namespace std;\n\ndouble StepsToMiles(int userSteps) {\n  return static_cast<double>(userSteps) / 2000.0;\n}\n\nint main() {\n  int steps;\n  cin >> steps;\n\n  double miles = StepsToMiles(steps);\n\n  cout << fixed << setprecision(4) << miles << endl;\n\n  return 0;\n}", "class": 1, "version": 3, "model": "Gemini", "variation": 9}
```

## Project Structure

```
├── app.py                      # Main Streamlit application with 7-tab interface
├── config.py                   # Configuration parameters
├── requirements.txt            # Python dependencies
│
├── Core Detection
│   ├── code_explainer.py       # TF-IDF based AI probability prediction
│   ├── gradient_explainer.py   # UniXcoder embedding-based detection
│   ├── codebert_explainer.py   # CodeBERT model utilities
│   └── region_authorship.py    # Mixed-authorship detection with baseline normalization
│
├── Model Training
│   ├── evaluate_models.py      # Model evaluation and metrics calculation
│   ├── model_trainer.py        # ML model training and ensemble creation
│   ├── data_loader.py          # JSONL data loader
│   └── tokenizer_utils.py      # Tokenization (TF-IDF & UniXcoder embeddings)
│
├── AI Code Generation
│   ├── gemini.py               # Google Gemini API integration for bulk generation
│   └── chatgpt.py              # OpenAI GPT-5.1 API integration
│
└── Data
    ├── random_ai_1680.jsonl    # Sample AI-generated code dataset
    └── random_human_1680.jsonl # Sample human-written code dataset
```

## Supported Models

* **Random Forest** - Ensemble decision tree classifier
* **Support Vector Machine (SVM)** - High-dimensional pattern recognition
* **XGBoost** - Gradient boosting with regularization
* **Multi-layer Perceptron (MLP)** - Neural network classifier
* **Voting Ensemble** - Combined predictions from all models for maximum accuracy

## Key Technologies

### Machine Learning
* **TF-IDF Vectorization** - Term frequency analysis for code patterns
* **UniXcoder Embeddings** - Pre-trained transformer model for code understanding
* **Sliding Window Analysis** - Line-by-line mixed-authorship detection
* **Baseline Normalization** - Boilerplate filtering to reduce false positives

### APIs
* **Google Gemini (gemini-2.0-flash-exp)** - AI code generation
* **OpenAI GPT-5.1** - Advanced code generation and variation

### Frontend
* **Streamlit** - Interactive web interface with real-time updates
* **Pandas** - Data manipulation and analysis
* **Plotly/Matplotlib** - Visualization of metrics and heatmaps

## Dependencies

Key Python libraries required:

* **streamlit** - Web application framework
* **scikit-learn** - Machine learning models
* **xgboost** - Gradient boosting
* **transformers** - UniXcoder model
* **torch** - PyTorch for deep learning
* **sctokenizer** - Code tokenization
* **google-genai** - Google Gemini API
* **openai** - OpenAI GPT API
* **pandas, numpy** - Data processing

Install dependencies easily:

```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Training Custom Models
1. Navigate to **Upload & Run** tab
2. Upload human code JSONL file
3. Upload AI-generated code JSONL file
4. Click **Train Models** to train both TF-IDF and UniXcoder models
5. View results in **Compare All Models** tab

### 2. Analyzing Code Snippets
1. Go to **Try a Code Snippet** tab
2. Paste your C/C++ code
3. Choose model (TF-IDF or UniXcoder)
4. View:
   - **AI Probability Score** - Overall likelihood of AI generation
   - **Max Probability** - Highest suspicious score
   - **Line-by-Line Heatmap** - Visual representation of suspicious regions
   - **Risk Assessment** - Flagged high-risk sections

### 3. Generating Datasets
**Using Gemini:**
1. Go to **Bulk Generate (Gemini)** tab
2. Enter API key
3. Input prompts (pipe-separated: `prompt1 | prompt2`)
4. Input instruction styles (pipe-separated)
5. Generate 10 variations per combination
6. Download JSONL/CSV files

**Using ChatGPT:**
1. Go to **Bulk Generate (ChatGPT)** tab
2. Enter OpenAI API key
3. Follow same process as Gemini

## Academic Use

This tool is designed for **academic integrity** in C/C++ programming courses:
- Detect potential AI-assisted submissions
- Identify mixed human/AI authorship
- Flag suspicious code regions for manual review
- **Not foolproof** - Use as an assistive tool with human judgment

## Contributing

Contributions are welcome! Please create a pull request or open an issue for any enhancements or bug fixes.

## License

MIT License - See LICENSE.md for details

This project is licensed under the MIT License. See the LICENSE.md file for details.
