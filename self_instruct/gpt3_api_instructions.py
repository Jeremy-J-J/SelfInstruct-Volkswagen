import json
import tqdm
import os
import random
from openai import OpenAI
import openai
from datetime import datetime
import argparse
import time
    
SYSTEM_PROMPT = """
You are an autonomous-driving scenario engineer.  
1. Read the user’s batch of N OpenSCENARIO v1.0 XML-generation tasks.  
2. Extract the “static template” and the “dynamic slots”:  
   - Static template:  
     “Please generate / Create an OpenScenario v1.0 XML file, Ensure / Pay attention to ”  
   - Dynamic slots:  
     ① actor selection – pick **one or several** from {pedestrian, ego-vehicle, NPC-vehicle, bicycle, motorcycle, truck}.  
        - If >1 actor or >1 category, list **each actor’s driving intention**: motion direction, exact km/h integer speed, and relative position (left/right, same/opposite lane, cut-in, cut-out, overtaking, distance ahead/behind).  
     ② initial speeds – **concrete km/h integers only**:  
        - pedestrian 1–5 km/h  
        - bicycle 10–20 km/h  
        - motorcycle 15–40 km/h  
        - ego / NPC passenger car 10–120 km/h  
        - truck 10–100 km/h  
     ③ trigger distance – **exact integer metres** 5–200 m  
     ④ test goal – collision, lane-change, obstacle-avoidance, etc.  
3. Keep the original sentence structure, wording, numbering style, and technical terms; only vary the dynamic slots to fabricate a plausible 9th task.  
4. Output **only** the new task line—no commentary, no labels, no extra formatting.
"""

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