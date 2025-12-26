# AI Code Detector

This project is a Streamlit web application designed to distinguish between human-written and AI-generated code snippets. It leverages TF-IDF and CodeBERT embeddings combined with various machine learning models to perform accurate classification.

## Features

* **Upload and Train:** Users can upload two JSONL files—one containing human-written code samples and another with AI-generated code samples. The system trains multiple machine learning models using TF-IDF vectorization and CodeBERT embeddings.
* **Model Comparison:** Compare performance metrics of various trained models (Random Forest, SVM, XGBoost, MLP, and Ensemble) easily through a clean, interactive interface.
* **Interactive Classification:** Real-time code snippet classification using either TF-IDF or CodeBERT models.

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

This command launches a browser interface with three tabs:

* **Upload & Run:** Upload JSONL files and train the models.
* **Compare All Models:** View and compare evaluation metrics.
* **Try a Code Snippet:** Classify individual code snippets interactively.

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
├── app.py                     # Streamlit frontend
├── evaluate_models.py         # Data loading and evaluation
├── model_trainer.py           # Model training and ensemble creation
├── tokenizer_utils.py         # Tokenization and embeddings (TF-IDF & CodeBERT)
├── data_loader.py             # JSONL data loader
├── config.py                  # Configuration parameters
├── requirements.txt           # Python dependencies
└── data                       # Directory for storing training datasets
```

## Supported Models

* **Random Forest**
* **Support Vector Machine (SVM)**
* **XGBoost**
* **Multi-layer Perceptron (MLP)**
* **Voting Ensemble (combination of all above models)**

## Dependencies

Key Python libraries required:

* Streamlit
* Scikit-learn
* XGBoost
* Transformers (CodeBERT)
* SCTokenizer (for code tokenization)
* Pandas
* NumPy

Install dependencies easily:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please create a pull request or open an issue for any enhancements or bug fixes.

## License

License

This project is licensed under the MIT License. See the LICENSE.md file for details.
