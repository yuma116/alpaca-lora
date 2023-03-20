from peft import PeftModel
#from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from time import time

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"


if False: #torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16)
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )







# # tokenizer = LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# # model = LLaMAForCausalLM.from_pretrained(
# model = LlamaForCausalLM.from_pretrained(
    # "decapoda-research/llama-7b-hf",
    # load_in_8bit=True,
    # device_map="auto",
# )
# model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"] #.cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Response:", output.split("### Response:")[1].strip())

command = input("Instruction: ")

time_s = time()
evaluate(command)
time_e = time()

evaluate_time = time_e - time_s
print("evaluate time")
print(evaluate_time)

