import json
import tqdm
import os
import random
from openai import OpenAI
import openai
from datetime import datetime
import argparse
import time

# SYSTEM_PROMPT = """
# You are a Function Call task generator.  
# 1. Read the user's batch of N Function Call data-generation tasks.  
# 2. Extract the "static template" and the "dynamic slots":  
#    - Static template: "Please generate a Function Call task data, requirements: "  
#    - Dynamic slots:  
#      ① function name – pick **exactly one** from {read_file, edit_file, delete_file, create_file, list_directory, grep_search, file_search}.  
#      ② function description and **all** arguments (mandatory & optional):  
#         1> read_file: Read the contents of a file.  
#            Args:  
#              - target_file (string, required) – path of the file to read  
#              - offset (integer, optional) – 1-indexed line to start reading  
#              - limit (integer, optional) – number of lines to read  
#              - should_read_entire_file (boolean, optional) – read whole file flag  
#              - agent (object, optional) – agent instance for permission checks  
#         2> edit_file: Edit a file according to instructions.  
#            Args:  
#              - target_file (string, required) – path to the file  
#              - instructions (string, required) – edit description  
#              - code_edit (dict|string, optional) – line-based edits {"1-5":"new"}  
#              - code_replace (string, optional) – full replacement content  
#              - agent (object, optional) – agent instance for permission checks  
#         3> delete_file: Delete a file.  
#            Args:  
#              - target_file (string, required) – path of the file to delete  
#              - agent (object, optional) – agent instance for permission checks  
#         4> create_file: Create a new file.  
#            Args:  
#              - file_path (string, required) – path where to create  
#              - content (string, required) – file content  
#              - agent (object, optional) – agent instance for permission checks  
#         5> list_directory: List directory contents.  
#            Args:  
#              - relative_workspace_path (string, required) – path to list  
#              - agent (object, optional) – agent instance for permission checks  
#         6> grep_search: Fast regex search inside files.  
#            Args:  
#              - query (string, required) – regex pattern to search  
#              - explanation (string, optional) – reason for search  
#              - case_sensitive (boolean, optional) – case sensitivity flag  
#              - include_pattern (string, optional) – glob files to include  
#              - exclude_pattern (string, optional) – glob files to exclude  
#              - agent (object, optional) – agent instance (unused)  
#         7> file_search: Fast fuzzy file search.  
#            Args:  
#              - query (string, required) – fuzzy filename to search  
#              - explanation (string, optional) – reason for search  
#              - agent (object, optional) – agent instance (unused)  
# 3. Fabricate a plausible 9th task by keeping the original sentence structure and wording; vary only the dynamic slots.  
# 4. Output **only** the new task line—no commentary, no labels, no extra formatting.  
# """

SYSTEM_PROMPT = '''
You are a Function Call task generator.
1. Read the user's batch of N Function Call data-generation tasks.
2. Extract the "static template" and the "dynamic slots":
    - Static template: "Please generate a Function Call task data, requirements: "
    - Dynamic slots:
        ① function name – pick exactly one from {read_file, edit_file, delete_file, create_file, list_directory, grep_search, file_search, run_terminal_command}.
        ② function description (use the description below to represent the selected function, do not list parameters):
            1> read_file: Read the contents of a file.
            2> edit_file: Edit a file according to instructions.
            3> delete_file: Delete a file.
            4> create_file: Create a new file.
            5> list_directory: List directory contents.
            6> grep_search: Fast regex search inside files.
            7> file_search: Fast fuzzy file search.
            8> run_terminal_command: Run a terminal command.
3. Fabricate a plausible task by keeping the original sentence structure and wording; vary only the dynamic slots (function name and function description).
4. Output only the new task line—no commentary, no labels, no extra formatting.
5. Do not include the specific function name in your response. Instead of merely repeating the function description, expand upon it naturally to create a more fluent and context-rich instruction.

Example_0: Please generate a function call task data, requirements: I'll help you create a DSL code file for the specified test scenario. Let me first explore the existing structure to understand how to properly create this OSC file.\n\nFirst, let me check what files exist in the rag_osc/standard_test_scenarios folder to understand the structure:\n
Example_1: Please generate a function call task data, requirements: Let me also check another example to get a better understanding of the structure:\n\n
Example_2: Please generate a function call task data, requirements: Now let me check the basic scenarios directory to see if there are any base templates I should use:\n\n
Example_3: Please generate a function call task data, requirements: Based on the requirements and examples I've seen, I'll now create the CDA_023.osc file that matches the specified test intent. Let me create this file with the proper structure:\n\n
Example_4: Please generate a function call task data, requirements: Let me check what other files might be relevant for lane changing scenarios:\n\n
Example_5: Please generate a function call task data, requirements: Now let me examine a few example files to understand the structure better, starting with CutInExit.osc since it seems most relevant to our cut-in scenario:\n\n
Example_6: Please generate a function call task data, requirements: Let me try to see what files contain \"extend test_config\":\n\n
'''

def make_requests(
        engine, prompts, max_tokens, temperature, top_p, 
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None, organization=None
    ):
    response = []
    target_length = max_tokens
    # if api_key is not None:
    #     openai.api_key = api_key
    # if organization is not None:
    #     openai.organization = organization
    
    client = OpenAI(api_key=api_key, base_url="http://localhost:8007/v1")
    # client = OpenAI(api_key=api_key, base_url="http://10.160.199.227:8006/v1")
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries:
        try:
            for p in prompts:
                resp = client.chat.completions.create(
                    model="holo-model",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
                    max_tokens=target_length,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop_sequences,
                    logprobs=logprobs,
                    n=1,
                )
                response.append(resp.choices[0].message.content)
            break
        except openai.APIStatusError as e:          # 1. 新基类
            print(f"OpenAIError: {e}")
            # 2. 新错误提示
            if "maximum context length" in str(e).lower():
                target_length = int(target_length * 0.8)
                print(f"Reducing target length to {target_length}, retrying...")
            # 3. 可单独处理限流
            elif isinstance(e, openai.RateLimitError):
                print(f"Rate limit hit, retrying in {backoff_time}s…")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 1.5, 60)   # 封顶 60 s
            else:
                print(f"Retrying in {backoff_time}s…")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 1.5, 60)
            retry_cnt += 1
    
    if isinstance(prompts, list):
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": response[j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)
        return results
    else:
        data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }
        return [data]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument(
        "--logprobs",
        default=5,
        type=int,
        help="The `logprobs` parameter of GPT3"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The `n` parameter of GPT3. The number of responses to generate."
    )
    parser.add_argument(
        "--best_of",
        type=int,
        help="The `best_of` parameter of GPT3. The beam size on the GPT3 server."
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()

    
if __name__ == "__main__":
    random.seed(123)
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompt = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i: i + args.request_batch_size]
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop_sequences=args.stop_sequences,
                    logprobs=args.logprobs,
                    n=args.n,
                    best_of=args.best_of,
                )
                for data in results:
                    fout.write(json.dumps(data) + "\n")