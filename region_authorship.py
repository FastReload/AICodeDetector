"""
Region-level / mixed-authorship analysis for code.

Provides per-line AI probability by scoring overlapping windows of lines,
similar to GPTZero's sentence-level detection.

This approach detects mixed authorship: some lines might be human-written
while others are AI-generated, which is common in real academic scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple
import html


@dataclass
class WindowResult:
    """Result from analyzing a single window of code lines."""
    start_line: int  # 0-based
    end_line: int    # inclusive, 0-based
    ai_prob: float
    snippet: str


@dataclass
class RegionAnalysisResult:
    """Complete analysis of code regions with per-line probabilities."""
    per_line_ai_prob: List[float]          # One probability per line
    windows: List[WindowResult]             # All analyzed windows
    avg_ai_prob: float                      # Overall average
    max_ai_prob: float                      # Most suspicious region
    most_suspicious_lines: List[Tuple[int, float, str]]  # (1-based line_no, prob, line_text)
    suspicious_regions: List[Tuple[int, int, float]]     # (start_line, end_line, avg_prob) 1-based


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _is_boilerplate_line(line: str) -> bool:
    """Check if a line is C/C++ boilerplate that should be marked as low-risk."""
    line_stripped = line.strip()
    line_lower = line_stripped.lower()
    
    # Empty lines or comments
    if not line_stripped or line_stripped.startswith('//') or line_stripped.startswith('/*'):
        return True
    
    # C/C++ include statements
    if line_stripped.startswith('#include') or 'include<' in line_stripped.replace(' ', ''):
        return True
    
    # C++ using namespace
    if 'using' in line_lower and 'namespace' in line_lower:
        return True
    
    # Just opening/closing braces
    if line_stripped in ['{', '}', '};']:
        return True
    
    return False


def analyze_mixed_authorship(
    code: str,
    predict_ai_proba: Callable[[str], float],
    *,
    window_lines: int = 10,
    stride_lines: int = 3,
    top_k: int = 10,
    region_threshold: float = 0.7,
) -> RegionAnalysisResult:
    """
    Analyze code for mixed authorship by sliding a window over lines.
    
    Args:
        code: Source code string
        predict_ai_proba: Function that returns P(AI) for a code snippet
        window_lines: Number of lines per window (default 10)
        stride_lines: Step size between windows (default 3)
        top_k: Number of most suspicious lines to return
        region_threshold: Probability threshold to flag a region as suspicious
        
    Returns:
        RegionAnalysisResult with per-line probabilities and suspicious regions
    """
    lines = code.splitlines()
    n = len(lines)

    if n == 0:
        return RegionAnalysisResult([], [], 0.0, 0.0, [], [])

    window_lines = max(1, int(window_lines))
    stride_lines = max(1, int(stride_lines))
    top_k = max(0, int(top_k))

    windows: List[WindowResult] = []
    per_line_sum = [0.0] * n
    per_line_cnt = [0] * n

    # Slide window over code lines
    start = 0
    while start < n:
        end = min(n - 1, start + window_lines - 1)
        snippet = "\n".join(lines[start : end + 1])
        
        # Get AI probability for this window
        if snippet.strip():
            try:
                p_ai = _clamp01(float(predict_ai_proba(snippet)))
            except Exception as e:
                print(f"Warning: prediction failed for window {start}-{end}: {e}")
                p_ai = 0.0
        else:
            p_ai = 0.0

        windows.append(WindowResult(
            start_line=start, 
            end_line=end, 
            ai_prob=p_ai,
            snippet=snippet[:200]  # Store first 200 chars for debugging
        ))

        # Accumulate probabilities for each line in this window
        for i in range(start, end + 1):
            per_line_sum[i] += p_ai
            per_line_cnt[i] += 1

        if end == n - 1:
            break
        start += stride_lines

    # Average probabilities for each line across all windows that included it
    per_line = [
        (per_line_sum[i] / per_line_cnt[i]) if per_line_cnt[i] else 0.0
        for i in range(n)
    ]
    
    # Calculate boilerplate baseline (noise floor) for this file
    boilerplate_probs = [per_line[i] for i in range(n) if _is_boilerplate_line(lines[i])]
    baseline = sum(boilerplate_probs) / len(boilerplate_probs) if boilerplate_probs else 0.0
    
    # Normalize probabilities relative to baseline
    # Keep boilerplate weight higher so it contributes to overall suspicion
    normalized_per_line = []
    for i in range(n):
        if _is_boilerplate_line(lines[i]):
            # Boilerplate lines: keep most of their original probability
            normalized_per_line.append(per_line[i] * 0.75)  # Keep 75% of original value
        else:
            # Actual code: keep original probability but amplify if above baseline
            if per_line[i] > baseline:
                # Amplify code that's more suspicious than baseline
                normalized = _clamp01(per_line[i] * 1.2)
            else:
                # Keep original if at or below baseline
                normalized = per_line[i]
            normalized_per_line.append(normalized)
    
    per_line = normalized_per_line

    avg_ai = sum(per_line) / n if n else 0.0
    max_ai = max(per_line) if per_line else 0.0

    # Find most suspicious individual lines
    ranked = sorted(
        [(i, per_line[i], lines[i]) for i in range(n)],
        key=lambda t: t[1],
        reverse=True,
    )
    most = [(i + 1, p, txt) for (i, p, txt) in ranked[:top_k]]

    # Identify suspicious regions (consecutive lines above threshold)
    suspicious_regions = []
    region_start = None
    region_probs = []
    
    for i, p in enumerate(per_line):
        if p >= region_threshold:
            if region_start is None:
                region_start = i
                region_probs = [p]
            else:
                region_probs.append(p)
        else:
            if region_start is not None:
                # End of suspicious region
                avg_region_prob = sum(region_probs) / len(region_probs)
                suspicious_regions.append((
                    region_start + 1,  # 1-based
                    i,                 # 1-based (exclusive end)
                    avg_region_prob
                ))
                region_start = None
                region_probs = []
    
    # Handle region that extends to end of file
    if region_start is not None:
        avg_region_prob = sum(region_probs) / len(region_probs)
        suspicious_regions.append((
            region_start + 1,
            n,
            avg_region_prob
        ))

    return RegionAnalysisResult(
        per_line_ai_prob=per_line,
        windows=windows,
        avg_ai_prob=avg_ai,
        max_ai_prob=max_ai,
        most_suspicious_lines=most,
        suspicious_regions=suspicious_regions
    )


def render_line_heatmap_html(
    code: str,
    per_line_ai_prob: List[float],
    *,
    show_line_numbers: bool = True,
    threshold: float = 0.7,
) -> str:
    """
    Render code with per-line background highlighting (like GPTZero).
    
    Args:
        code: Source code
        per_line_ai_prob: AI probability for each line
        show_line_numbers: Whether to show line numbers
        threshold: Probability threshold for "high risk" styling
        
    Returns:
        HTML string with styled code
    """
    lines = code.splitlines()
    n = len(lines)
    probs = (per_line_ai_prob[:n] + [0.0] * max(0, n - len(per_line_ai_prob))) if n else []

    def bg(p: float) -> str:
        """Generate background color based on probability (stronger contrast)."""
        p = _clamp01(p)
        # Gradient: transparent (low) -> yellow (medium) -> red (high)
        if p < 0.3:
            # Low risk: subtle gray
            alpha = 0.06 + 0.16 * (p / 0.3)
            return f"rgba(120, 120, 120, {alpha:.3f})"
        elif p < 0.6:
            # Medium-low risk: brighter yellow
            intensity = (p - 0.3) / 0.3
            alpha = 0.30 + 0.25 * intensity
            return f"rgba(255, 220, 80, {alpha:.3f})"
        elif p < 0.8:
            # Medium-high risk: bold orange
            intensity = (p - 0.6) / 0.2
            alpha = 0.55 + 0.25 * intensity
            return f"rgba(255, 150, 50, {alpha:.3f})"
        else:
            # High risk: strong red
            intensity = (p - 0.8) / 0.2
            alpha = 0.75 + 0.20 * intensity
            return f"rgba(255, 40, 40, {alpha:.3f})"

    def border_left(p: float) -> str:
        """Add left border for high-risk lines (stronger signal)."""
        if p >= 0.8:
            return "border-left: 6px solid #ff2222;"
        elif p >= 0.6:
            return "border-left: 5px solid #ff7a1a;"
        elif p >= 0.4:
            return "border-left: 4px solid #ffcc33;"
        return "border-left: 3px solid rgba(120,120,120,0.25);"

    rows = []
    for idx, line in enumerate(lines, start=1):
        p = probs[idx - 1]
        ln = f"{idx:>4}  " if show_line_numbers else ""
        safe = html.escape(line).replace(" ", "&nbsp;").replace("\t", "&nbsp;" * 4)
        
        # Risk indicator
        if p >= 0.8:
            risk_icon = "🔴"
            risk_label = "Very High AI Probability"
        elif p >= 0.7:
            risk_icon = "🟠"
            risk_label = "High AI Probability"
        elif p >= 0.5:
            risk_icon = "🟡"
            risk_label = "Medium AI Probability"
        elif p >= 0.3:
            risk_icon = "⚪"
            risk_label = "Low-Medium AI Probability"
        else:
            risk_icon = ""
            risk_label = "Low AI Probability"
        
        title_text = f"{risk_label} - Line {idx}: P(AI)={p:.1%}"

        rows.append(
            f'<div style="white-space:pre; font-family:Consolas,Monaco,monospace; '
            f'font-size:13px; padding:5px 12px; background:{bg(p)}; {border_left(p)} '
            f'transition: all 0.2s; margin:0;" '
            f'title="{html.escape(title_text)}">'
            f'<span style="color:#888; user-select:none; font-weight:500;">{ln}</span>'
            f'<span style="color:#e0e0e0;">{safe}</span>'
            f'<span style="float:right; font-size:12px; margin-left:10px; user-select:none;">{risk_icon}</span>'
            f'</div>'
        )

    return (
        '<div style="border:1px solid #3a3a3a; border-radius:8px; overflow:hidden; '
        'box-shadow: 0 2px 8px rgba(0,0,0,0.1);">'
        + "".join(rows)
        + "</div>"
    )


def format_suspicious_regions_text(regions: List[Tuple[int, int, float]]) -> str:
    """
    Format suspicious regions as readable text.
    
    Args:
        regions: List of (start_line, end_line, avg_prob) tuples
        
    Returns:
        Formatted string describing suspicious regions
    """
    if not regions:
        return "✅ No highly suspicious regions detected (all lines < 70% AI probability)"
    
    lines = ["⚠️ **Suspicious Regions Detected:**\n"]
    for i, (start, end, prob) in enumerate(regions, 1):
        if start == end - 1:
            lines.append(f"{i}. **Line {start}**: {prob:.1%} AI probability")
        else:
            lines.append(f"{i}. **Lines {start}-{end-1}**: {prob:.1%} average AI probability")
    
    return "\n".join(lines)


def calculate_optimal_window_size(num_lines: int) -> int:
    """
    Automatically calculate optimal window size based on code length.
    
    Args:
        num_lines: Number of lines in the code
        
    Returns:
        Optimal window size in lines
    """
    if num_lines <= 20:
        return min(8, num_lines)
    elif num_lines <= 50:
        return 10
    elif num_lines <= 100:
        return 12
    elif num_lines <= 200:
        return 15
    else:
        return 20


def generate_risk_assessment(per_line_probs: List[float], avg_ai_prob: float) -> Tuple[str, str, str]:
    """
    Generate risk assessment message based on line probabilities.
    
    Args:
        per_line_probs: List of AI probabilities per line
        avg_ai_prob: Overall average AI probability
        
    Returns:
        Tuple of (risk_level, emoji, message)
    """
    if not per_line_probs:
        return "low", "✅", "No code to analyze"
    
    red_lines = sum(1 for p in per_line_probs if p >= 0.75)
    yellow_lines = sum(1 for p in per_line_probs if 0.4 <= p < 0.75)
    total_lines = len(per_line_probs)
    
    red_pct = red_lines / total_lines if total_lines else 0
    yellow_pct = yellow_lines / total_lines if total_lines else 0
    
    # High risk: any significant red presence
    if red_lines >= 5 or red_pct >= 0.25:
        return (
            "high",
            "🔴",
            f"**HIGH RISK**: {red_lines} red lines ({red_pct:.0%}) show strong AI signature. "
            "Code likely contains AI-generated sections. **Immediate discussion required.**"
        )
    
    # Medium-high: some red or heavy yellow
    if red_lines >= 2 or (yellow_pct >= 0.5 and yellow_lines >= 10):
        return (
            "medium-high",
            "🟠",
            f"**MEDIUM-HIGH RISK**: {red_lines} red lines, {yellow_lines} yellow lines. "
            "**Context and explanation needed** for highlighted sections."
        )
    
    # Medium: notable yellow presence
    if yellow_pct >= 0.35 or yellow_lines >= 6:
        return (
            "medium",
            "🟡",
            f"**DISCUSSION NEEDED**: {yellow_lines} lines ({yellow_pct:.0%}) show moderate AI signature. "
            "Ask for context or walkthrough of these parts."
        )
    
    # Low-medium: some yellow, elevated average
    if yellow_lines >= 3 or avg_ai_prob >= 0.35:
        return (
            "low-medium",
            "⚪",
            f"**LOW-MEDIUM RISK**: {yellow_lines} lines show AI-like patterns. "
            "Brief discussion recommended to confirm authorship."
        )
    
    # Low risk
    return (
        "low",
        "✅",
        f"**LOW RISK**: Code appears mostly human-written. "
        f"Only {yellow_lines + red_lines} flagged lines out of {total_lines}."
    )
