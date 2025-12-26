"""
Code Explainer Module - TF-IDF Based Prediction
Provides AI probability predictions for code snippets using TF-IDF vectorization
"""

from sctokenizer import CppTokenizer


class CodeExplainer:
    """
    Provides AI detection predictions using TF-IDF and traditional ML classifiers
    """
    
    def __init__(self, model, vectorizer, class_names=['Human', 'AI']):
        """
        Args:
            model: Trained classifier (e.g., Random Forest, XGBoost)
            vectorizer: Fitted TfidfVectorizer
            class_names: Labels for the classes
        """
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names
        self.cpp_tokenizer = CppTokenizer()
    
    def tokenize_code(self, code):
        """Tokenize code using sctokenizer"""
        tokens = self.cpp_tokenizer.tokenize(code)
        token_values = [token.token_value for token in tokens]
        return ' '.join(token_values)
    
    def predict_ai_probability(self, code: str) -> float:
        """
        Returns P(AI) for a code snippet using the TF-IDF model.
        Used for region-level analysis.
        
        Args:
            code: Source code string
            
        Returns:
            Probability that code is AI-generated (0.0 to 1.0)
        """
        tokenized_code = self.tokenize_code(code)
        X = self.vectorizer.transform([tokenized_code])
        proba = self.model.predict_proba(X)[0]
        # Assumes class order: [Human, AI]
        return float(proba[1])


def create_explainer(model, vectorizer):
    """
    Factory function to create a CodeExplainer
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TfidfVectorizer
        
    Returns:
        CodeExplainer instance
    """
    return CodeExplainer(model, vectorizer)
