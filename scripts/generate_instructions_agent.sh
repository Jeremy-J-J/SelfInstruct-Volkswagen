batch_dir=data_agent/gpt3_generations/

python self_instruct/bootstrap_instructions_agent.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 20 \
    --seed_tasks_path seed/seed_tasks_openscenario_agent.jsonl \
    --engine "davinci"