batch_dir=data_v3_qwen3-coder-next/gpt3_generations/

python self_instruct/identify_clf_or_not.py \
    --batch_dir ${batch_dir} \
    --engine "davinci" \
    --request_batch_size 5