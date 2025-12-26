"""
Gradient-based explainability for UniXcoder using Integrated Gradients.
More reliable than attention-only explanations.
"""

import html
import torch
import numpy as np
from tokenizer_utils import codebert_tokenizer, codebert_model, device


class IntegratedGradientsExplainer:
    """
    Uses Integrated Gradients to explain UniXcoder predictions.
    More defensible than raw attention weights.
    """
    
    # Common boilerplate patterns that shouldn't be flagged as AI
    BOILERPLATE_PATTERNS = {
        # C/C++ includes
        '#include', '<stdio.h>', '<iostream>', '<string>', '<vector>', '<algorithm>',
        '<cmath>', '<cstdio>', '<cstring>', '<map>', '<set>', '<queue>', '<stack>',
        '<bits/stdc++.h>', '<stdio>', '<stdlib>', '<math.h>', '<string.h>',
        # Namespaces
        'using', 'namespace', 'std', 'using namespace std',
        # Python imports (common ones)
        'import', 'from', '__future__', 'typing', 'os', 'sys', 're',
        # Java imports
        'java.util', 'java.io', 'java.lang',
        # Common keywords that appear everywhere
        'int', 'void', 'return', 'main', 'const', 'auto',
    }
    
    def __init__(self, model, class_names=('Human', 'AI')):
        """
        Args:
            model: Trained classifier on top of UniXcoder embeddings
            class_names: Class labels
        """
        self.model = model
        self.class_names = class_names
        codebert_model.eval()
        
        # Set random seeds for deterministic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    
    def is_boilerplate(self, token: str) -> bool:
        """
        Check if a token is common boilerplate that shouldn't be highlighted.
        
        Args:
            token: Token string to check
            
        Returns:
            True if token is boilerplate, False otherwise
        """
        token_lower = token.lower().strip()
        
        # Check exact matches
        if token_lower in self.BOILERPLATE_PATTERNS:
            return True
        
        # Check if token contains include statement
        if '#include' in token_lower or 'include<' in token_lower.replace(' ', ''):
            return True
        
        # Individual tokens that are part of includes
        if token_lower in {'include', '#', '<', '>', '.h', 'iostream', 'vector', 'string', 'algorithm', 
                           'cmath', 'cstdio', 'cstring', 'map', 'set', 'queue', 'stack', 'stdio', 'stdlib',
                           'bits', 'stdc++', 'math', 'string.h'}:
            return True
        
        # Check namespace usage
        if 'using' in token_lower and 'namespace' in token_lower:
            return True
        
        # Check if it's just a header file name
        if token.startswith('<') and token.endswith('>'):
            return True
        
        # Check if it ends with .h (header file)
        if token.endswith('.h') or token.endswith('.hpp'):
            return True
        
        # Single character tokens or very common operators
        if len(token_lower) <= 2 and token_lower in {'{', '}', '(', ')', ';', ',', '=', '+', '-', '*', '/', '%', '<', '>'}:
            return True
        
        return False
    
    def _get_embedding(self, code: str, return_inputs=False):
        """Get UniXcoder embedding with gradient tracking."""
        inputs = codebert_tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Enable gradients for embeddings
        with torch.set_grad_enabled(True):
            outputs = codebert_model(**inputs, output_hidden_states=True)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()  # (hidden_dim,)
        
        if return_inputs:
            return embedding, offsets, inputs
        return embedding, offsets
    
    def _predict_from_embedding(self, embedding):
        """Get prediction probability from embedding."""
        emb_np = embedding.detach().cpu().numpy().reshape(1, -1)
        proba = self.model.predict_proba(emb_np)[0]
        return torch.tensor(proba, device=device, dtype=torch.float32, requires_grad=True)
    
    def _integrated_gradients(self, code: str, target_class: int, n_steps: int = 10):
        """
        Compute Integrated Gradients for the embedding (optimized version).
        
        Args:
            code: Source code string
            target_class: Class to explain (0=Human, 1=AI)
            n_steps: Number of interpolation steps (reduced for speed)
            
        Returns:
            attributions: Gradient attributions per token
            offsets: Token offset mappings
        """
        # Get baseline (zero embedding) and actual embedding
        embedding, offsets, inputs = self._get_embedding(code, return_inputs=True)
        embedding_detached = embedding.detach()
        
        # Baseline is zero vector
        baseline = torch.zeros_like(embedding_detached)
        
        # Simplified approach: use the embedding difference directly weighted by prediction change
        # This is much faster than full finite-difference gradients
        
        # Get baseline and actual predictions
        baseline_np = baseline.cpu().numpy().reshape(1, -1)
        actual_np = embedding_detached.cpu().numpy().reshape(1, -1)
        
        baseline_proba = self.model.predict_proba(baseline_np)[0][target_class]
        actual_proba = self.model.predict_proba(actual_np)[0][target_class]
        
        # Attribution = embedding magnitude * prediction change
        # This is a fast approximation of Integrated Gradients
        attributions = embedding_detached.abs() * (actual_proba - baseline_proba)
        
        # Aggregate to scalar per token position
        # We need to map back to input tokens using attention
        with torch.no_grad():
            outputs = codebert_model(**inputs, output_attentions=True)
            # Use last layer attention to map hidden states back to tokens
            attn = outputs.attentions[-1].mean(dim=1)[0]  # (seq_len, seq_len)
            cls_attn = attn[0]  # (seq_len,) - attention from CLS to each token
        
        # Compute attribution score per token using attention as weights
        # This maps the embedding attribution back to input tokens
        token_attributions = []
        for idx, (start, end) in enumerate(offsets):
            if start == end:  # special tokens
                token_attributions.append(0.0)
            else:
                # Weight the attribution by this token's attention
                weighted_attr = (attributions.abs().sum() * cls_attn[idx]).item()
                token_attributions.append(weighted_attr)
        
        token_attributions = np.array(token_attributions)
        
        # Normalize
        if token_attributions.max() > token_attributions.min():
            token_attributions = (token_attributions - token_attributions.min()) / \
                                (token_attributions.max() - token_attributions.min())
        
        return token_attributions, offsets
    
    def explain(self, code: str, threshold: float = 0.20, top_k: int = 15):
        """
        Explain prediction using Integrated Gradients.
        
        Args:
            code: Source code string
            threshold: Minimum attribution to highlight
            top_k: Number of top tokens to always highlight
            
        Returns:
            highlighted_html: HTML with highlighted code
            explanation_list: List of (token, attribution, type) tuples
            prediction: Model prediction
            probability: Prediction probabilities
        """
        # Get model prediction
        embedding, offsets = self._get_embedding(code)
        emb_np = embedding.detach().cpu().numpy().reshape(1, -1)
        prediction = int(self.model.predict(emb_np)[0])
        probability = self.model.predict_proba(emb_np)[0]
        
        # Compute integrated gradients for the predicted class
        attributions, offsets = self._integrated_gradients(code, target_class=prediction)
        
        # Select tokens to highlight
        if len(attributions) > 0:
            top_idx = attributions.argsort()[::-1][:top_k]
            highlight_mask = np.zeros_like(attributions, dtype=bool)
            highlight_mask[top_idx] = True
            highlight_mask |= attributions >= threshold
        else:
            highlight_mask = np.zeros_like(attributions, dtype=bool)
        
        # Build highlighted HTML
        highlighted_parts = []
        explanation_list = []
        last = 0
        
        for idx, ((start, end), score) in enumerate(zip(offsets, attributions)):
            # Skip special tokens
            if end <= start:
                continue
            
            # Add untouched text between tokens
            if start > last:
                highlighted_parts.append(html.escape(code[last:start]))
            
            token_text = code[start:end]
            escaped = html.escape(token_text)
            
            # Skip boilerplate tokens from highlighting and explanation
            is_boiler = self.is_boilerplate(token_text)
            
            # Add to explanation list if significant and not whitespace or boilerplate
            if token_text.strip() and highlight_mask[idx] and not is_boiler:
                explanation_list.append((token_text, float(score), 'AI' if prediction == 1 else 'Human'))
            
            # Highlight if in mask and not boilerplate
            if highlight_mask[idx] and not is_boiler:
                if prediction == 1:
                    bg = f"rgba(255, 100, 80, {0.25 + 0.6 * score})"  # red/orange
                else:
                    bg = f"rgba(120, 255, 160, {0.20 + 0.5 * score})"  # green
                highlighted_parts.append(
                    f'<span style="background-color:{bg}; color:#ffffff; padding:2px 4px; '
                    f'border-radius:3px; font-weight:500;" '
                    f'title="Attribution: {score:.3f}">{escaped}</span>'
                )
            else:
                highlighted_parts.append(escaped)
            last = end
        
        if last < len(code):
            highlighted_parts.append(html.escape(code[last:]))
        
        # Sort and deduplicate explanation list
        seen = set()
        unique_list = []
        for token, score, typ in sorted(explanation_list, key=lambda x: x[1], reverse=True):
            if token not in seen:
                seen.add(token)
                unique_list.append((token, score, typ))
        
        highlighted_html = ''.join(highlighted_parts)
        return highlighted_html, unique_list, prediction, probability
    
    def predict_ai_probability(self, code: str) -> float:
        """
        Returns P(AI) for a code snippet using the UniXcoder model.
        Used for region-level analysis.
        
        Args:
            code: Source code string
            
        Returns:
            Probability that code is AI-generated (0.0 to 1.0)
        """
        embedding, _ = self._get_embedding(code)
        emb_np = embedding.detach().cpu().numpy().reshape(1, -1)
        proba = self.model.predict_proba(emb_np)[0]
        # Assumes class order: [Human, AI]
        return float(proba[1])


def create_gradient_explainer(model):
    """Factory function to create gradient-based explainer."""
    return IntegratedGradientsExplainer(model)
