import os
import json
import csv
import time
import google.generativeai as genai

gemini_model = None

def set_gemini_api_key(api_key):
    global gemini_model
    try:
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
        )
        return True
    except Exception as e:
        return False

def ask_model(prompt):
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text.replace("```cpp\n", "").replace("```", "").strip()

def generate_bulk_gemini_codes(prompts_str, versions_str, progress_callback=None):
    prompts = [p.strip() for p in prompts_str.split('|') if p.strip()]
    versions = [v.strip() for v in versions_str.split('|') if v.strip()]
    output_jsonl = "gemini_generated.jsonl"
    output_csv = "gemini_generated.csv"

    with open(output_jsonl, "w") as jsonl_file, open(output_csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["problem", "code", "class", "version", "model", "variation"])
        total = len(prompts) * len(versions) * 10
        counter = 0

        for i, prompt in enumerate(prompts, start=1):
            for j, version in enumerate(versions, start=1):
                base_prompt = version + "\n" + prompt

                for variation in range(1, 11):
                    try:
                        cpp_code = ask_model(base_prompt)
                    except Exception as e:
                        cpp_code = f"ERROR: {e}"

                    json_record = {
                        "problem": i,
                        "code": cpp_code,
                        "class": 1,
                        "version": j,
                        "model": "Gemini",
                        "variation": variation
                    }

                    jsonl_file.write(json.dumps(json_record) + "\n")
                    csv_writer.writerow([i, cpp_code, 1, j, "Gemini", variation])

                    counter += 1
                    if progress_callback:
                        progress_callback(counter / total)

                    time.sleep(3)

    return output_jsonl, output_csv
