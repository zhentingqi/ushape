from transformers import AutoTokenizer
import transformers
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


ckpt = "meta-llama/Llama-2-7b-hf"
    
tokenizer = LlamaTokenizer.from_pretrained(ckpt)
model = LlamaForCausalLM.from_pretrained(ckpt)

input_sequence = "I've been waiting for a HuggingFace course my whole life. The sentiment of this sentence is: "
model_inputs = tokenizer([input_sequence], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs)
output_sequence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(output_sequence)