import json
import argparse
import os
import random
import openai

def translate_to_chinese(text, api_base, model_id):
    """
    使用大模型服务将文本翻译成中文
    """
    client = openai.OpenAI(
        base_url=api_base,
        api_key="EMPTY"
    )
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "请将以下内容翻译成中文，不要有任何额外的解释或发挥，只返回翻译结果。"},
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"翻译出错: {e}")
        return text  # 如果翻译失败，返回原文


def convert_data(input_file_path, output_file_path, translate_ratio=1/3):
    """
    从输入文件中读取JSONL格式的数据，筛选出type为GenXML的记录，
    并将其转换为指定格式的JSONL数据，然后写入输出文件。
    其中一部分数据的prompt会被翻译成中文。
    """
    
    # 系统消息内容
    system_message_content = (
        "You are an expert in interpreting the intent of autonomous-driving "
        "OpenScenario v1.0 XML scenario description files; please analyze the "
        "input XML and understand its autonomous-driving intent."
    )
    
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        index = 0
        translated_count = 0
        
        for line in infile:
            # 解析每一行的JSON数据
            data = json.loads(line.strip())
            
            # 检查type字段是否为GenXML
            if data.get('type') == 'GenXML':
                # 根据索引决定是否翻译（大约每三次取一次进行翻译）
                should_translate = (index % 3 == 0)
                
                prompt_content = data['prompt']
                
                if should_translate:
                    print(f"正在翻译第 {index} 条数据的prompt...")
                    prompt_content = translate_to_chinese(data['prompt'], 
                                                         "http://localhost:8007/v1", 
                                                         "holo-model")
                    translated_count += 1
                
                # 创建新的数据格式
                converted_data = {
                    "messages": [
                        {
                            "content": system_message_content,
                            "role": "system"
                        },
                        {
                            "content": data['completion'],  # completion字段的值放入user角色
                            "role": "user"
                        },
                        {
                            "content": prompt_content,     # 可能是翻译后的prompt字段的值放入assistant角色
                            "role": "assistant"
                        }
                    ]
                }
                
                # 写入输出文件
                outfile.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
                
            index += 1
    
    print(f"总共翻译了 {translated_count} 条数据")


def main():
    parser = argparse.ArgumentParser(description='Convert GenXML records from JSONL to a new format.')
    parser.add_argument(
        '--input_file',
        type=str,
        default='/workspace/pro/selfInstruct/SelfInstruct-Volkswagen/data_10k/gpt3_generations/finetuning_data/gpt3_finetuning_data_4168.jsonl',
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./converted_genxmls.jsonl',
        help='Output JSONL file path'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        return
    
    # 执行数据转换
    print(f"Converting data from {args.input_file} to {args.output_file}")
    convert_data(args.input_file, args.output_file)
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()