import json
import os

BASE_DIR = "/workspace/pro/selfInstruct/SelfInstruct-Volkswagen/data/gpt3_generations/"
def extract_completions_to_xosc():
    # 定义输入和输出路径
    input_file = BASE_DIR +"finetuning_data/gpt3_finetuning_data_102.jsonl"
    output_dir = BASE_DIR + "xosc"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return
    
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 读取jsonl文件并处理每一行
    count = 0
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if line:
                try:
                    # 解析JSON行
                    data = json.loads(line)
                    
                    # 提取completion字段
                    completion = data.get("completion", "")
                    
                    # 生成输出文件名（四位数字格式）
                    count += 1
                    output_filename = f"{count:04d}.xosc"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 写入文件
                    with open(output_path, 'w', encoding='utf-8') as outfile:
                        outfile.write(completion)
                    
                    print(f"已保存第 {count} 个completion到 {output_filename}")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                except Exception as e:
                    print(f"处理过程中出现错误: {e}")
    
    print(f"处理完成！共保存了 {count} 个xosc文件到 {output_dir}")

if __name__ == "__main__":
    extract_completions_to_xosc()