import pdb
import torch
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
from transformers import BloomForCausalLM 
from transformers import LlamaForCausalLM


def batch_inference(ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    if "opt" in ckpt:
        model = OPTForCausalLM.from_pretrained(ckpt).to(device)
    elif "bloomz" in ckpt:
        model = BloomForCausalLM.from_pretrained(ckpt).to(device)
    elif "llama" in ckpt:
        model = LlamaForCausalLM.from_pretrained(ckpt).to(device)

    prompt = ["Please say something about yourself: I am ", "How are you doing? Did you pass the math exam? I"]
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
    print(inputs)

    # Generate
    generate_ids = model.generate(**inputs)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(outputs)


if __name__ == "__main__":
    ckpt = "facebook/opt-6.7b"
    # ckpt = "bigscience/bloomz-7b1"
    # ckpt = "/root/autodl-fs/llama-to-hf"
    batch_inference(ckpt)
