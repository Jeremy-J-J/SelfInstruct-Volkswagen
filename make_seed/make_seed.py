import json 
import copy 
import os 
from gpt3_api_instructions import make_requests, parse_args
import random 

args = parse_args()

fout = open("make_seed/seed_tasks_openscenario_intention2osc_en.jsonl", "w")
all_dicts = []
with open("make_seed/seed_tasks_openscenario_intention2xsoc_en.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        data_2 = copy.deepcopy(data)
        data_2['id'] = data_2['id'] + "_2"
        data_2['name'] = "openscenario_generation_osc_en"
        osc_path = os.path.join('make_seed', 'seedosc', data['id'] + '.osc')
        with open(osc_path, "r") as fosc:
            osc_content = fosc.read()
        data_2['instances'][0]['output'] = osc_content

        results = make_requests(
                    engine=args.engine,
                    prompts=[data['instruction']],
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
        data_2['instruction'] = results[0]['response']
        all_dicts.append(data_2)

        print(results[0]['response'])
        
random.shuffle(all_dicts)
for data in all_dicts:
    fout.write(json.dumps(data) + "\n")