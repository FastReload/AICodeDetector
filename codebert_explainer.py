"""
CodeBERT explanation using attention weights to highlight influential tokens.
"""

import html
import torch
import numpy as np
from tokenizer_utils import codebert_tokenizer, codebert_model, device, get_codebert_embedding


class CodeBertExplainer:
    def __init__(self, model, class_names=('Human', 'AI')):
        self.model = model
        self.class_names = class_names

    def _get_attention_scores(self, code: str):
        """Return offsets and normalized attention scores per token.

        We aggregate attention across all layers and heads, focusing on CLS -> token
        attention, and drop special tokens (offset start == end == 0).
        """
        inputs = codebert_tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        offsets = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = codebert_model(**inputs, output_attentions=True)

        # Stack attentions: (layers, batch, heads, seq, seq)
        attn_layers = torch.stack(outputs.attentions)  # (L, B, H, S, S)
        # Mean over layers and heads, take batch 0
        attn_mean = attn_layers.mean(dim=(0, 2))[0]  # (S, S)
        cls_attn = attn_mean[0]  # attention from CLS to others (S,)

        token_scores = []
        for (start, end), score in zip(offsets, cls_attn.tolist()):
            if start == end:  # special tokens like <s>, </s>
                token_scores.append(0.0)
            else:
                token_scores.append(score)

        token_scores = np.array(token_scores, dtype=float)
        # Normalize to 0-1
        if token_scores.max() > token_scores.min():
            token_scores = (token_scores - token_scores.min()) / (token_scores.max() - token_scores.min())
        return offsets, token_scores

    def explain(self, code: str, threshold: float = 0.15, top_k: int = 15):
        # Model prediction and probability
        emb = get_codebert_embedding(code).reshape(1, -1)
        prediction = int(self.model.predict(emb)[0])
        prob = self.model.predict_proba(emb)[0]

        offsets, scores = self._get_attention_scores(code)

        # Select tokens to highlight: above threshold OR in top-k
        if len(scores) > 0:
            top_idx = scores.argsort()[::-1][:top_k]
            highlight_mask = np.zeros_like(scores, dtype=bool)
            highlight_mask[top_idx] = True
            highlight_mask |= scores >= threshold
        else:
            highlight_mask = np.zeros_like(scores, dtype=bool)

        highlighted_parts = []
        explanation_list = []
        last = 0

        for idx, ((start, end), score) in enumerate(zip(offsets, scores)):
            # Skip special tokens (start == end == 0)
            if end <= start:
                continue
                
            # Add untouched text between tokens
            if start > last:
                highlighted_parts.append(html.escape(code[last:start]))
            
            token_text = code[start:end]
            escaped = html.escape(token_text)
            
            # Skip whitespace-only tokens from explanation list
            if token_text.strip() and highlight_mask[idx]:
                explanation_list.append((token_text, float(score), 'AI' if prediction == 1 else 'Human-like'))

            if highlight_mask[idx]:
                if prediction == 1:
                    bg = f"rgba(255, 100, 80, {0.25 + 0.5 * score})"  # red/orange
                else:
                    bg = f"rgba(120, 255, 160, {0.20 + 0.4 * score})"  # green
                highlighted_parts.append(
                    f'<span style="background-color:{bg}; color:#ffffff; padding:2px 4px; border-radius:3px;" '
                    f'title="Weight: {score:.3f}">{escaped}</span>'
                )
            else:
                highlighted_parts.append(escaped)
            last = end

        if last < len(code):
            highlighted_parts.append(html.escape(code[last:]))

        # Sort explanation list by weight and remove duplicates
        seen = set()
        unique_list = []
        for token, weight, typ in sorted(explanation_list, key=lambda x: x[1], reverse=True):
            if token not in seen:
                seen.add(token)
                unique_list.append((token, weight, typ))
        explanation_list = unique_list

        highlighted_html = ''.join(highlighted_parts)
        return highlighted_html, explanation_list, prediction, prob


def create_codebert_explainer(model):
    return CodeBertExplainer(model)
