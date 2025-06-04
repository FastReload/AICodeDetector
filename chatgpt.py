import json
import csv
import time
import openai

chatgpt_model_name = "gpt-4o"
chatgpt_client = None

def set_chatgpt_api_key(api_key):
    global chatgpt_client
    try:
        chatgpt_client = openai
        chatgpt_client.api_key = api_key
        return True
    except Exception as e:
        return False

def generate_chatgpt_cpp_code(prompt):
    if not chatgpt_client or not chatgpt_client.api_key:
        return "Error: API key not set. Please enter your OpenAI API key first."

    try:
        response = chatgpt_client.chat.completions.create(
            model=chatgpt_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only outputs C++ code. Only output code. Always start with #include."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=1024,
        )
        text = response.choices[0].message.content
        return text.replace("```cpp\n", "").replace("```", "").strip()
    except Exception as e:
        return f"Error generating code: {e}"

# NEW FUNCTION
def generate_bulk_chatgpt_codes(prompts_str, versions_str, progress_callback=None):
    if not chatgpt_client or not chatgpt_client.api_key:
        raise RuntimeError("OpenAI API key not set.")

    prompts = [p.strip() for p in prompts_str.split('|') if p.strip()]
    versions = [v.strip() for v in versions_str.split('|') if v.strip()]
    total = len(prompts) * len(versions) * 10
    counter = 0

    jsonl_path = "gpt4_generated.jsonl"
    csv_path = "gpt4_generated.csv"

    with open(jsonl_path, "w") as jsonl_file, open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["problem", "code", "class", "version", "model", "variation"])

        for i, prompt in enumerate(prompts, start=1):
            for j, version in enumerate(versions, start=1):
                full_prompt = version + "\n" + prompt

                for variation in range(1, 11):
                    try:
                        response = chatgpt_client.chat.completions.create(
                            model=chatgpt_model_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that only outputs C++ code. Output only code. Start with #include. No explanation."},
                                {"role": "user", "content": full_prompt}
                            ],
                            temperature=0.8,
                            top_p=0.95,
                            frequency_penalty=0,
                            presence_penalty=0,
                            max_tokens=1024,
                        )
                        code = response.choices[0].message.content
                        code = code.replace("```cpp\n", "").replace("```", "").strip()
                    except Exception as e:
                        code = f"ERROR: {e}"

                    row = {
                        "problem": i,
                        "code": code,
                        "class": 1,
                        "version": j,
                        "model": "GPT-4o",
                        "variation": variation
                    }

                    jsonl_file.write(json.dumps(row) + "\n")
                    writer.writerow([i, code, 1, j, "GPT-4o", variation])

                    counter += 1
                    if progress_callback:
                        progress_callback(counter / total)

                    time.sleep(2)

    return jsonl_path, csv_path
