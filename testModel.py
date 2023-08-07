import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
import os
import platform

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。相关使用指引，请见examples/tokenizer_showcase.ipynb
path = "G:/code/Qwen-7B"

# quantization configuration for NF4 (4 bits)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
model = AutoModelForCausalLM.from_pretrained(path,
                                             quantization_config=quantization_config,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             bf16=True).eval()

model.generation_config = GenerationConfig.from_pretrained(path)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
history = []
print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
while True:
    query = input("\n用户：")
    if query == "stop":
        break
    if query == "clear":
        history = []
        os.system(clear_command)
        print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
        continue
    response, history = model.chat(tokenizer=tokenizer, query=query, history=history)
    os.system(clear_command)
    print(response, flush=True)
