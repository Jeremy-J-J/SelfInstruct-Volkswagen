batch_dir=data_agent/gpt3_generations/

python self_instruct/prepare_for_finetuning_agent.py \
    --instance_files ${batch_dir}/machine_generated_instances.jsonl \
    --classification_type_files ${batch_dir}/is_clf_or_not_davinci_template_1.jsonl \
    --output_dir ${batch_dir}/finetuning_data \
    --include_seed_tasks \
    --seed_tasks_path seed/seed_tasks_openscenario_agent.jsonl 