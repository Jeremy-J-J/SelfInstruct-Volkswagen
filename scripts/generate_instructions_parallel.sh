batch_dir=data_v2/gpt3_generations/

python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 20000 \
    --seed_tasks_path seed/seed_tasks_openscenario_v2.jsonl \
    --engine "davinci"