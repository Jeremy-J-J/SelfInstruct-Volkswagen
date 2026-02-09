import json
import os
import openai
from openai import OpenAI


def summarize_prompt(client, prompt):
    """
    使用大模型服务对prompt进行总结
    """
    try:
        response = client.chat.completions.create(
            model="holo-model",
            messages=[
                {"role": "user", "content": f"Please provide a concise summary of the following task in English without expanding on it or providing additional explanations: {prompt}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用大模型服务出错: {e}")
        return prompt


def convert_data_format(input_file, output_file):
    """
    转换数据格式
    """
    client = OpenAI(base_url="http://localhost:8007/v1", api_key="dummy-key")

    # 固定的tools定义 - 修改为JSON对象数组而不是字符串
    tools_definition = [
        {"type": "function", "function": {"name": "read_file", "description": "Read the contents of a file.", "parameters": {"type": "object", "properties": {"target_file": {"type": "string", "description": "The path of the file to read"}, "offset": {"type": "integer", "description": "The line number to start reading from (1-indexed)"}, "limit": {"type": "integer", "description": "The number of lines to read"}, "should_read_entire_file": {"type": "boolean", "description": "Whether to read the entire file"}}, "required": ["target_file"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Edit a file in the codebase.", "parameters": {"type": "object", "properties": {"target_file": {"type": "string", "description": "The target file to modify"}, "instructions": {"type": "string", "description": "A single sentence instruction describing the edit"}, "code_edit": {"type": ["object", "string", "null"], "description": "Line-based edit with line ranges as keys (e.g., \\\"1-5\\\") and values as the new content.", "additionalProperties": {"type": "string", "description": "New content for the specified line range"}, "examples": [{"1-5": "def new_function():\\n    return True", "10-15": "# This is a multi-line\\n# comment block"}]}, "code_replace": {"type": ["string", "null"], "description": "Complete replacement content for the file (use this instead of code_edit for full file replacement)"}}, "required": ["target_file", "instructions"]}}},
        {"type": "function", "function": {"name": "delete_file", "description": "Delete a file at the specified path.", "parameters": {"type": "object", "properties": {"target_file": {"type": "string", "description": "The path of the file to delete"}}, "required": ["target_file"]}}},
        {"type": "function", "function": {"name": "create_file", "description": "Create a new file with the given content.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Path where the file should be created"}, "content": {"type": "string", "description": "Content to write to the file"}}, "required": ["file_path", "content"]}}},
        {"type": "function", "function": {"name": "list_directory", "description": "List the contents of a directory.", "parameters": {"type": "object", "properties": {"relative_workspace_path": {"type": "string", "description": "Path to list contents of"}}, "required": ["relative_workspace_path"]}}},
        {"type": "function", "function": {"name": "run_terminal_command", "description": "Run a terminal command. IMPORTANT: Always use non-interactive flags that works for the command you are running (like --yes, -y, --no-interaction, yes | , --quiet, or equivalent). i.e git commit -m 'commit message' --no-interaction or yes | npx create-next-app@latest --no-interactive", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The terminal command to execute. MUST be non-interactive and should include all necessary flags to prevent or bypass prompts."}, "explanation": {"type": "string", "description": "Explanation of why this command needs to be run"}, "is_background": {"type": "boolean", "description": "Whether to run in the background"}, "require_user_approval": {"type": "boolean", "description": "Whether user approval is required"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "codebase_search", "description": "Search the codebase using semantic search.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query to find relevant code"}, "target_directories": {"type": "array", "items": {"type": "string"}, "description": "Directories to search over"}, "explanation": {"type": "string", "description": "One sentence explanation as to why this tool is being used"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "grep_search", "description": "Fast text-based search using regex patterns.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The regex pattern to search for", "explanation": "One sentence explanation as to why this tool is being used"}, "case_sensitive": {"type": "boolean", "description": "Whether the search is case sensitive"}, "include_pattern": {"type": "string", "description": "Glob pattern for files to include"}, "exclude_pattern": {"type": "string", "description": "Glob pattern for files to exclude"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "file_search", "description": "Fast file search based on fuzzy matching against file path.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Fuzzy filename to search for"}, "explanation": {"type": "string", "description": "One sentence explanation as to why this tool is being used"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "trend_search", "description": "Search for trending topics related to a query", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The query to search trends for"}, "explanation": {"type": "string", "description": "Optional explanation of why this search is being performed"}, "country_code": {"type": "string", "description": "Country code for trends (default: US)"}, "days": {"type": "integer", "description": "Number of days to look back for trends (default: 7)"}, "max_results": {"type": "integer", "description": "Maximum number of trends to return (default: 3)"}, "lookback_hours": {"type": "integer", "description": "Number of hours to look back for Google Trends data (default: 48)"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "query_images", "description": "Query an AI model about one or more images.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The question or query about the image(s)"}, "image_paths": {"type": "array", "items": {"type": "string"}, "description": "List of local paths to image files to analyze"}}, "required": ["query", "image_paths"]}}}
    ]
    
    # 用户消息的模板
    user_message_template = """\nYou are a powerful agentic AI coding assistant. You operate in a coding environment.\n\nYou are pair programming with a USER to solve their coding task.\nThe task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.\nEach time the USER sends a message, we automatically attach information about their current state, such as what files they have open, where their cursor is, recently viewed files, edit history in their session so far, and more.\nThis information may be relevant to the coding task, so consider it carefully.\nYour main goal is to follow the USER's instructions at each message.\n\n<tool_calling>\nYou have tools at your disposal to solve the coding task. Follow these rules regarding tool calls:\n1. ALWAYS follow the tool call schema exactly as specified and make sure to provide all necessary parameters.\n2. NEVER call tools that are not explicitly provided.\n3. Only call tools when they are necessary. If the USER's task is general or you already know the answer, just respond without calling tools.\n4. Before calling each tool, explain to the USER why you are calling it.\n</tool_calling>\n\n<making_code_changes>\nWhen making code changes, NEVER output code to the USER, unless requested. Instead use the code edit tools to implement the change.\nIt is *EXTREMELY* important that your generated code can be run immediately by the USER. To ensure this, follow these instructions carefully:\n1. Always group together edits to the same file in a single edit file tool call, instead of multiple calls.\n2. If you're creating the codebase from scratch, create an appropriate dependency management file with package versions and a helpful README.\n3. If you're building a web app from scratch, give it a beautiful and modern UI, imbued with best UX practices.\n4. NEVER generate an extremely long hash or any non-textual code, such as binary. These are not helpful to the USER.\n5. Unless you are appending some small edit to a file, or creating a new file, you MUST read the contents or section of what you're editing before editing it.\n6. If you've introduced errors, fix them if clear how to. Do not make uneducated guesses.\n</making_code_changes>\n\n<searching_and_reading>\nYou have tools to search the codebase and read files. Follow these rules regarding tool calls:\n1. If available, prefer the semantic search tool to grep search, file search, and list dir tools.\n2. If you need to read a file, prefer to read larger sections of the file at once over multiple smaller calls.\n3. If you have found a reasonable place to edit or answer, do not continue calling tools.\n</searching_and_reading>\n\nYou MUST use the following format when citing code regions or blocks:\n``12:15:app/components/Todo.tsx\n```\nThis is the ONLY acceptable format for code citations. The format is ```startLine:endLine:filepath where startLine and endLine are line numbers.\n\nWhen breaking down complex tasks, consider each step carefully and use the appropriate tools for each part of the problem. Work iteratively toward a complete solution.\nWhen Task is Complete, respond with: task is complete or completed it successfully\n\nTasks:\n"""

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # 解析当前行的JSON数据
                data = json.loads(line)
                prompt = data.get("prompt", "")
                completion = data.get("completion", "")
                
                # 使用大模型服务对prompt进行总结
                summarized_task = summarize_prompt(client, prompt)
                
                new_data = {
                    "messages": [
                        {
                            "content": user_message_template + summarized_task,
                            "role": "user"
                        },
                        {
                            "content": prompt,
                            "role": "assistant"
                        },
                        {
                            "content": completion,
                            "role": "tool_call"
                        }
                    ],
                    "tools": tools_definition
                }
                
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
                print(f"已处理第 {line_num} 行数据")
                
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"处理第 {line_num} 行时发生错误: {e}")
                continue


if __name__ == "__main__":
    input_file = "/workspace/pro/selfInstruct/SelfInstruct-Volkswagen/data_agent/gpt3_generations/finetuning_data/gpt3_finetuning_data_test.jsonl" 
    output_file = "/workspace/pro/selfInstruct/SelfInstruct-Volkswagen/data_agent/gpt3_generations/finetuning_data/gpt3_finetuning_data_test_converted.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        exit(1)
    
    print("开始转换数据格式...")
    convert_data_format(input_file, output_file)
    print("数据格式转换完成!")