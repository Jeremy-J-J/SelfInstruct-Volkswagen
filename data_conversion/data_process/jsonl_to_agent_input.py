import json
import requests
import time
import os
import uuid
from tqdm import tqdm

def call_gpt_api(prompt_text):
    """
    调用GPT API，获取精简总结
    """
    url = "http://localhost:8007/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-11111111111111111111111111111111"  # 使用默认密钥
    }
    
    payload = {
        "model": "holo-model",
        "messages": [
            {
                "role": "system",
                "content": "请用英文提炼一下这句话，不要发散，不要杂话连篇"
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # 提取AI的回复内容
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        # 如果API调用失败，返回原始prompt作为降级处理
        return prompt_text


def count_lines(filename):
    """
    计算文件总行数
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def process_jsonl_to_agent_format(input_file_path, output_file_path):
    """
    将JSONL文件转换为代理输入格式
    """
    total_lines = count_lines(input_file_path)
    processed_count = 0
    
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        # 创建进度条
        with tqdm(total=total_lines, desc="处理数据", unit="line") as pbar:
            for line_num, line in enumerate(infile, 1):
                try:
                    # 解析每一行的JSON数据
                    data = json.loads(line.strip())
                    
                    # 提取prompt和completion中的content
                    prompt = data.get("prompt", "")
                    completion_args = data.get("completion", {}).get("arguments", {})
                    content = completion_args.get("content", "")
                    
                    # 调用GPT API生成instruction
                    instruction = call_gpt_api(prompt)
                    
                    # 创建数据模板
                    template = {
                        "id": f"generated_{line_num}",
                        "name": "auto_generated_task",
                        "instruction": instruction,
                        "instances": [
                            {
                                "input": content,
                                "output": prompt
                            }
                        ],
                        "is_classification": False
                    }
                    
                    # 写入输出文件
                    outfile.write(json.dumps(template, ensure_ascii=False) + "\n")
                    
                    processed_count += 1
                    
                    # 添加小延迟以避免API调用过于频繁
                    time.sleep(0.1)
                    
                except json.JSONDecodeError:
                    # 不增加计数，但仍然更新进度条
                    pass
                except Exception as e:
                    # 不增加计数，但仍然更新进度条
                    pass
                
                # 更新进度条
                pbar.set_postfix({"Processed": processed_count})
                pbar.update(1)
                    
    print(f"处理完成！共处理了 {processed_count} 条有效数据。")


if __name__ == "__main__":
    input_file = "/workspace/pro/selfInstruct/SelfInstruct-Volkswagen/data_conversion/data_process/data_create_file_function_call_747.jsonl"
    output_file = "/workspace/pro/selfInstruct/SelfInstruct-Volkswagen/data_conversion/data_process/agent_input_data.jsonl"
    
    print("开始处理数据...")
    process_jsonl_to_agent_format(input_file, output_file)
    print("数据处理完成！")