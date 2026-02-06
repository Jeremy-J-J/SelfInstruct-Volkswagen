import json
import argparse
import os

def convert_data(input_file_path, output_file_path):
    """
    从输入文件中读取JSONL格式的数据，筛选出type为GenXML的记录，
    并将其转换为指定格式的JSONL数据，然后写入输出文件。
    """
    
    # 系统消息内容
    system_message_content = (
        "You are an expert in interpreting the intent of autonomous-driving "
        "OpenScenario v1.0 XML scenario description files; please analyze the "
        "input XML and understand its autonomous-driving intent."
    )
    
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # 解析每一行的JSON数据
            data = json.loads(line.strip())
            
            # 检查type字段是否为GenXML
            if data.get('type') == 'GenXML':
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
                            "content": data['prompt'],     # prompt字段的值放入assistant角色
                            "role": "assistant"
                        }
                    ]
                }
                
                # 写入输出文件
                outfile.write(json.dumps(converted_data, ensure_ascii=False) + '\n')

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