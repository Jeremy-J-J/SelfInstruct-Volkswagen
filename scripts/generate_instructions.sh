batch_dir=data/gpt3_generations/

python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 100 \
    --seed_tasks_path seed/seed_tasks_openscenario.jsonl \
    --engine "davinci"