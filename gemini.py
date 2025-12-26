# To run this code you need to install the following dependencies:
# pip install google-genai

import os
import json
import csv
import time

from google import genai
from google.genai import types

# ---------------------------------------------------
# 1. Configuration & API Key
# ---------------------------------------------------

# Paste your Google AI Studio API key here
DIRECT_API_KEY = ""

# ---------------------------------------------------
# 2. Prompts and style versions
# ---------------------------------------------------
prompts = [
    "Write a function MaxMagnitude with two input parameters, that returns the largest magnitude value. If the inputs are 5 7, the function returns 7. If the inputs are -8 -2, the function returns -8. Use the function in a program that takes two integer inputs and outputs the largest magnitude value.",
    "A pedometer treats walking 2,000 steps as walking 1 mile. Write a program whose input is the number of steps and whose output is the miles walked. Your program must define and call a function double StepsToMiles(int userSteps).",
    "An acronym is a word formed from the initial letters of words in a set phrase. Write a program that returns an acronym. Lowercase starting words should not count. Define a function string CreateAcronym(string userPhrase).",
    "Write a program whose input is a character and a string and outputs how many times the character appears. Case matters. Define a function int CountCharacters(char userChar, string userString).",
    "A leap year has 366 days. A year is a leap year if divisible by 4 but century years must also be divisible by 400. Define a function bool IsLeapYear(int userYear).",
    "Write a program that takes N word pairs: name and phone number. Then read a name and print the phone number. Define string GetPhoneNumber(vector<string> nameVec, vector<string> phoneNumberVec, string contactName)."
]

versions = [
    "Write the code in cpp. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp like an introductory programmer. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp like an undergraduate CS1 student. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp like a beginner first year programmer. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp. Only use basic syntax to avoid AI detection. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp. Make sure that the code cannot be detected as AI generated. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp. Use basic fundamental concepts only. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp as a human would with inconsistencies and unpolished comments. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp as if it was a real student final attempt. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp in an unpolished way. It should be mildly inefficient and should not use advanced libraries or techniques. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp using only the most basic libraries. Do not include anything extra. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp like an introductory programmer and also ensure to use libraries and coding style a very basic programmer would. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp like an undergraduate CS1 student and use only the necessary libraries and use very basic structure. Do not include explanations or examples. Only output the code file content.",
    "Write the code in cpp like an undergraduate CS1 student and use only the necessary libraries and use a very very primitive structure, just like how a novice would think. Do not include explanations or examples. Only output the code file content."
]

# ---------------------------------------------------
# 3. Helper: generate one C++ solution (no streaming)
# ---------------------------------------------------
def generate_cpp(client, model, gen_config, prompt_text):
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt_text,
            config=gen_config,
        )
    except Exception as e:
        print("Gemini error:", e)
        return "ERROR_CALL_FAILED"

    text = response.text or ""

    # Strip code fences if any
    text = text.replace("```cpp", "").replace("```c++", "")
    text = text.replace("```", "").strip()

    # Heuristic: keep only the code part starting at first code marker
    markers = ["#include", "int main", "using namespace std"]
    start_idx = None
    for m in markers:
        if m in text:
            idx = text.find(m)
            if start_idx is None or idx < start_idx:
                start_idx = idx

    if start_idx is not None:
        text = text[start_idx:].strip()

    if not text:
        return "ERROR_EMPTY_OUTPUT"

    return text


# ---------------------------------------------------
# 4. Functions for Streamlit app integration
# ---------------------------------------------------
def set_gemini_api_key(api_key):
    """Sets the API key globally for use in generation"""
    global DIRECT_API_KEY
    DIRECT_API_KEY = api_key
    return True


def generate_bulk_gemini_codes(prompts_str, versions_str, progress_callback=None):
    """
    Generate bulk C++ codes using the new google.genai API.
    
    Args:
        prompts_str: |-separated prompts
        versions_str: |-separated instruction styles
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (jsonl_path, csv_path)
    """
    global DIRECT_API_KEY
    
    if not DIRECT_API_KEY or DIRECT_API_KEY == "":
        raise ValueError("API key not set. Call set_gemini_api_key() first.")
    
    # Parse input strings
    prompts_list = [p.strip() for p in prompts_str.split('|') if p.strip()]
    versions_list = [v.strip() for v in versions_str.split('|') if v.strip()]
    
    # Initialize client
    client = genai.Client(api_key=DIRECT_API_KEY)
    model = "gemini-3-pro-preview"
    
    gen_config = types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.95,
        max_output_tokens=1024,
    )
    
    jsonl_path = "gemini_generated.jsonl"
    csv_path = "gemini_generated.csv"
    
    with open(jsonl_path, "w") as jsonl_file, open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["problem", "code", "class", "version", "model", "variation"])
        
        total_samples = len(prompts_list) * len(versions_list) * 10
        counter = 0
        
        for i, problem_prompt in enumerate(prompts_list, start=1):
            for j, version_prompt in enumerate(versions_list, start=1):
                base_prompt = version_prompt + "\n" + problem_prompt
                
                for variation in range(1, 11):  # 10 variations
                    counter += 1
                    cpp_code = generate_cpp(client, model, gen_config, base_prompt)
                    
                    if cpp_code.startswith("ERROR"):
                        print("Error at problem {}, version {}, variation {}: {}".format(
                            i, j, variation, cpp_code
                        ))
                    
                    json_record = {
                        "problem": i,
                        "code": cpp_code,
                        "class": 1,
                        "version": j,
                        "model": model,
                        "variation": variation,
                    }
                    
                    jsonl_file.write(json.dumps(json_record) + "\n")
                    csv_writer.writerow([i, cpp_code, 1, j, model, variation])
                    
                    if progress_callback:
                        progress_callback(counter / total_samples)
                    
                    time.sleep(1)  # Rate limiting
    
    return jsonl_path, csv_path


# ---------------------------------------------------
# 5. Main generation loop (for standalone use)
# ---------------------------------------------------
def main():
    api_key = DIRECT_API_KEY

    if not api_key or api_key == "":
        raise ValueError("Please paste your API key into DIRECT_API_KEY at the top of the script.")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash-exp"

    gen_config = types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.95,
        max_output_tokens=1024,
    )

    print("Running a quick benchmark on one sample...")
    test_prompt = versions[0] + "\n" + prompts[0]
    t0 = time.time()
    sample_code = generate_cpp(client, model, gen_config, test_prompt)
    t1 = time.time()
    print("One sample took {:.2f} seconds on this Gemini model.".format(t1 - t0))
    # Optional: inspect the first sample
    print(sample_code)

    total_samples = len(prompts) * len(versions) * 10
    print("Starting full generation for {} samples...".format(total_samples))

    jsonl_path = "cpp_gemini_840codes.jsonl"
    csv_path = "cpp_gemini_840codes.csv"

    with open(jsonl_path, "w") as jsonl_file, open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["problem", "code", "class", "version", "model", "variation"])

        counter = 0
        for i, problem_prompt in enumerate(prompts, start=1):
            for j, version_prompt in enumerate(versions, start=1):
                base_prompt = version_prompt + "\n" + problem_prompt

                for variation in range(1, 11):  # 10 variations
                    counter += 1
                    cpp_code = generate_cpp(client, model, gen_config, base_prompt)

                    if cpp_code.startswith("ERROR"):
                        print("Error at problem {}, version {}, variation {}: {}".format(
                            i, j, variation, cpp_code
                        ))

                    json_record = {
                        "problem": i,
                        "code": cpp_code,
                        "class": 1,
                        "version": j,
                        "model": model,
                        "variation": variation,
                    }

                    jsonl_file.write(json.dumps(json_record) + "\n")
                    csv_writer.writerow([i, cpp_code, 1, j, model, variation])

                    if counter % 20 == 0:
                        print("Generated {}/{} samples...".format(counter, total_samples))

    print("Done. Files saved as:")
    print(jsonl_path)
    print(csv_path)


if __name__ == "__main__":
    main()
