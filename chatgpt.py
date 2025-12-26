import os
import json
import csv
import time
from openai import OpenAI

# ---------------------------------------------------
# 1. Configuration & API Key
# ---------------------------------------------------
API_KEY = ""
model_name = "gpt-5.1"

client = None

# ---------------------------------------------------
# 2. Functions for Streamlit app integration
# ---------------------------------------------------
def set_chatgpt_api_key(api_key):
    """Sets the API key globally for use in generation"""
    global API_KEY, client
    try:
        API_KEY = api_key
        client = OpenAI(api_key=API_KEY)
        # Test the API key with a simple request
        test_response = client.models.list()
        return True
    except Exception as e:
        print(f"API Key validation error: {e}")
        client = None
        API_KEY = ""
        return False


def generate_chatgpt_cpp_code(prompt):
    """Generate a single C++ code snippet (used by app.py)"""
    if not client or not API_KEY:
        return "Error: API key not set. Please enter your OpenAI API key first."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only outputs C++ code. Only output code. Always start with #include."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            max_completion_tokens=1024,
        )
        text = response.choices[0].message.content
        return text.replace("```cpp\n", "").replace("```c++", "").replace("```", "").strip()
    except Exception as e:
        return f"Error generating code: {e}"


def generate_bulk_chatgpt_codes(prompts_str, versions_str, progress_callback=None):
    """
    Generate bulk C++ codes using OpenAI API.
    
    Args:
        prompts_str: |-separated prompts
        versions_str: |-separated instruction styles
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (jsonl_path, csv_path)
    """
    if not client or not API_KEY:
        raise RuntimeError("OpenAI API key not set.")

    prompts_list = [p.strip() for p in prompts_str.split('|') if p.strip()]
    versions_list = [v.strip() for v in versions_str.split('|') if v.strip()]
    total = len(prompts_list) * len(versions_list) * 10
    counter = 0

    jsonl_path = "gpt5_generated.jsonl"
    csv_path = "gpt5_generated.csv"

    with open(jsonl_path, "w") as jsonl_file, open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["problem", "code", "class", "version", "model", "variation"])

        for i, prompt in enumerate(prompts_list, start=1):
            for j, version in enumerate(versions_list, start=1):
                full_prompt = version + "\n" + prompt

                for variation in range(1, 11):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that only outputs C++ code. Output only code. Start with #include. No explanation."},
                                {"role": "user", "content": full_prompt}
                            ],
                            temperature=0.8,
                            top_p=0.95,
                            frequency_penalty=0,
                            presence_penalty=0,
                            max_completion_tokens=1024,
                        )
                        code = response.choices[0].message.content
                        code = code.replace("```cpp\n", "").replace("```c++", "").replace("```", "").strip()
                    except Exception as e:
                        code = f"ERROR: {e}"

                    row = {
                        "problem": i,
                        "code": code,
                        "class": 1,
                        "version": j,
                        "model": model_name,
                        "variation": variation
                    }

                    jsonl_file.write(json.dumps(row) + "\n")
                    writer.writerow([i, code, 1, j, model_name, variation])

                    counter += 1
                    if progress_callback:
                        progress_callback(counter / total)

                    time.sleep(1)

    return jsonl_path, csv_path


# ---------------------------------------------------
# 3. Helper functions for JSONL fixing (requires prompts/versions)
# ---------------------------------------------------
def is_incomplete(code: str) -> bool:
    if not isinstance(code, str):
        return True

    stripped = code.strip()

    if stripped == "":
        return True

    if stripped.startswith("ERROR"):
        return True

    # Heuristic: code should end with proper closure
    if not stripped.endswith(("}", ");", "}", "]", ")")):
        return True

    return False


def ask_model(prompt_text: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only outputs C++ code. Output only code. Start with #include. No explanation."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.8,
        top_p=0.95,
        max_completion_tokens=4000
    )

    text = response.choices[0].message.content or ""
    text = text.replace("```cpp", "").replace("```c++", "").replace("```", "").strip()
    return text


def scan_existing(path):
    """Scan once to find seen keys and incomplete keys."""
    seen_keys = set()
    bad_keys = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print("Skipping JSON error on line", line_no, ":", e)
                continue

            try:
                key = (obj["problem"], obj["version"], obj["variation"])
            except KeyError as e:
                print("Skipping missing key on line", line_no, ":", e)
                continue

            seen_keys.add(key)

            code = obj.get("code", "")
            if is_incomplete(code):
                bad_keys.add(key)

    return seen_keys, bad_keys


def compute_expected_keys(num_prompts, num_versions):
    """Compute expected keys based on number of prompts and versions"""
    expected = set()
    for p in range(1, num_prompts + 1):
        for v in range(1, num_versions + 1):
            for var in range(1, 11):
                expected.add((p, v, var))
    return expected


def write_good_records(JSONL_IN, JSONL_OUT, num_prompts, num_versions):
    """Write all good existing records from input to output."""
    seen_keys, bad_keys = scan_existing(JSONL_IN)
    expected_keys = compute_expected_keys(num_prompts, num_versions)

    missing_keys = expected_keys - seen_keys
    to_regen = bad_keys | missing_keys

    print("Total expected keys:", len(expected_keys))
    print("Existing keys      :", len(seen_keys))
    print("Incomplete keys    :", len(bad_keys))
    print("Missing keys       :", len(missing_keys))
    print("Total to regenerate:", len(to_regen))

    # First pass for writing: copy only good records to the new file
    with open(JSONL_IN, "r", encoding="utf-8") as fin, open(
        JSONL_OUT, "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Already reported in scan, just skip here
                continue

            try:
                key = (obj["problem"], obj["version"], obj["variation"])
            except KeyError:
                continue

            # Skip bad ones, they will be regenerated
            if key in bad_keys:
                continue

            # Good record, copy as is
            fout.write(raw + "\n")

    return to_regen


def regenerate_records(to_regen, JSONL_OUT, prompts_list, versions_list):
    """Append regenerated records to the output file.
    
    Args:
        to_regen: Set of (problem, version, variation) tuples to regenerate
        JSONL_OUT: Output file path
        prompts_list: List of problem prompts
        versions_list: List of instruction style prompts
    """
    if not to_regen:
        return

    with open(JSONL_OUT, "a", encoding="utf-8") as fout:
        for problem, version_idx, variation in sorted(to_regen):
            problem_prompt = prompts_list[problem - 1]
            version_prompt = versions_list[version_idx - 1]
            base_prompt = version_prompt + "\n" + problem_prompt

            print(
                "Regenerating problem",
                problem,
                "version",
                version_idx,
                "variation",
                variation,
            )

            try:
                code = ask_model(base_prompt)
            except Exception as e:
                print(
                    "Error during regeneration for p{p}, v{v}, var{var}: {err}".format(
                        p=problem, v=version_idx, var=variation, err=e
                    )
                )
                code = "ERROR_CALL_FAILED"

            rec = {
                "problem": problem,
                "code": code,
                "class": 1,
                "version": version_idx,
                "model": model_name,
                "variation": variation,
            }

            fout.write(json.dumps(rec) + "\n")

            time.sleep(1.0)


# ---------------------------------------------------
# 4. Main function for standalone JSONL fixing
# ---------------------------------------------------
def main(prompts_list=None, versions_list=None):
    """Standalone script to fix incomplete JSONL files
    
    Args:
        prompts_list: List of problem prompts (required)
        versions_list: List of instruction style prompts (required)
    """
    global client
    
    if not API_KEY or API_KEY == "":
        raise ValueError("Please set API_KEY at the top of the script.")
    
    if not prompts_list or not versions_list:
        raise ValueError("prompts_list and versions_list are required. Import from gemini.py or define them.")
    
    client = OpenAI(api_key=API_KEY)
    
    safe_model_name = model_name.replace(".", "_")
    JSONL_IN = "cpp_code_10C_{name}.jsonl".format(name=safe_model_name)
    JSONL_OUT = "cpp_code_10C_{name}_fixed.jsonl".format(name=safe_model_name)
    
    if not os.path.exists(JSONL_IN):
        raise FileNotFoundError(
            "Input JSONL file not found: {path}".format(path=JSONL_IN)
        )

    to_regen = write_good_records(JSONL_IN, JSONL_OUT, len(prompts_list), len(versions_list))
    regenerate_records(to_regen, JSONL_OUT, prompts_list, versions_list)

    print("Wrote fixed JSONL:", JSONL_OUT)


if __name__ == "__main__":
    # To use standalone, import prompts and versions from gemini.py
    # Example:
    # from gemini import prompts, versions
    # main(prompts, versions)
    print("To use standalone mode, import prompts and versions:")
    print("  from gemini import prompts, versions")
    print("  main(prompts, versions)")
