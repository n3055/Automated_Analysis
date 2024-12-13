import os
import sys
import json
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get("AIPROXY_TOKEN")
if not api_key:
    raise EnvironmentError("AIPROXY_TOKEN not found in environment variables.")

# Ensure dataset is provided
if len(sys.argv) < 2:
    raise ValueError("Please provide a dataset file as a command-line argument.")

dataset = sys.argv[1]
try:
    df = pd.read_csv(dataset, encoding="ISO-8859-1")
except Exception as e:
    raise FileNotFoundError(f"Failed to read the dataset: {e}")

# Ensure outputs are saved in the same directory as the dataset
output_dir = os.path.dirname(os.path.abspath(dataset))

# Helper function for API calls
def call_ai_proxy(messages):
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise ConnectionError(f"API call failed with status {response.status_code}: {response.text}")
    return json.loads(response.text)["choices"][0]["message"]["content"]

# Step 1: Ask for recommended analyses
columns_info = f"Columns: {list(df.columns)}\nExample Rows:\n{df.iloc[:4].to_string(index=False)}"
prompt_analysis = (
    f"Given the following dataset information:\n{columns_info}\n"
    "Recommend High-Quality and insightful analyses that can be implemented in Python. Provide storytelling summaries and visuals using Seaborn."
    " Do not provide Python code; just describe the analyses."
)

messages = [{"role": "user", "content": prompt_analysis}]
try:
    analysis_recommendations = call_ai_proxy(messages)
except Exception as e:
    raise RuntimeError(f"Failed to retrieve analysis recommendations: {e}")

# Step 2: Request Python code for the recommended analyses
prompt_code = (
    f"Based on the recommended analyses {analysis_recommendations}, provide Python code to perform the analysis on the dataset '{dataset}'. "
    "Remove outliers and clean the data before performing the analysis such that output is of high quality and insightful.\n"
    "The script should save the following outputs:\n"
    "1. A Markdown file (README.md) summarizing the results as a story in a unique way that isn't boring. It should be well-structured and readable.\n"
    "2. 1-3 PNG charts as visualizations embedded in the README.md (using Seaborn plots), they should look visually appealing.\n"
    "In readme.md file you should narrate a story The data you received, briefly"
    "The analysis you carried out"
    "The insights you discovered"
    "The implications of your findings (i.e. what to do with the insights)\n"
    "Output only the Python code as a string that can be executed using the exec() function.\n"
    "Don't use ```python at the starting and ending of the output as it would not execute in the exec() function.\n"
    "Use encoding=\"ISO-8859-1\" while reading csv and also while writing to a file in binary mode.\n"
    "Don't include any prefixed variables like Generated Python Code:media_analysis_code = , your output should consist entirely of Python code.\n"
    "Don't address me or use English in your output, output only **PYTHON** without a Python code block."
)

messages.append({"role": "user", "content": prompt_code})
try:
    python_code = call_ai_proxy(messages)
    # Remove ```python or similar markers if present
    python_code = python_code.strip().lstrip('```python').rstrip('```')
except Exception as e:
    raise RuntimeError(f"Failed to retrieve Python code: {e}")

# Step 3: Execute the generated Python code
try:
    exec(python_code)
except Exception as e:
    print(f"Error executing generated code: {e}")
    feedback_prompt = (
        f"The code failed with the following error:\n{e}\n"
        "Please provide corrected Python code that fixes the issue. Output only the corrected Python code."
    )
    messages.append({"role": "user", "content": feedback_prompt})
    try:
        corrected_code = call_ai_proxy(messages)
        # Remove ```python or similar markers if present
        corrected_code = corrected_code.strip().lstrip('```python').rstrip('```')
        exec(corrected_code)
    except Exception as final_error:
        raise RuntimeError(f"Execution failed again: {final_error}")
