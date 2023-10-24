import pdb
import os
import math
import torch
from transformers import AutoTokenizer
from transformers import OPTForCausalLM, BloomForCausalLM, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
from xopen import xopen
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict):
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


def get_qa_prompt(
    question: str, documents: List[Document], mention_random_ordering: bool, query_aware_contextualization: bool
):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    if mention_random_ordering and query_aware_contextualization:
        raise ValueError("Mentioning random ordering cannot be currently used with query aware contextualization")

    if mention_random_ordering:
        prompt_filename = "qa_ordered_randomly.prompt"
    elif query_aware_contextualization:
        prompt_filename = "qa_with_query_aware_contextualization.prompt"
    else:
        prompt_filename = "qa.prompt"

    with open(prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def prepare_prompts(input_path, max_num_examples_to_get=200):
    prompts, examples, all_model_documents = [], [], []
    print("Reading data file...")
    with xopen(input_path) as fin:
        num_examples = 0
        for line in fin:
            input_example = json.loads(line)

            question = input_example["question"]
            documents = [Document.from_dict(ctx) for ctx in deepcopy(input_example["ctxs"])]
            prompt = get_qa_prompt(question, documents, mention_random_ordering=False, query_aware_contextualization=False)

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)
            
            num_examples += 1
            if num_examples == max_num_examples_to_get:
                break

    return prompts, examples, all_model_documents


def prepare_model_and_tokenizer(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "llama":
        model_path = "/root/autodl-fs/llama2-7b-to-hf"
        model = LlamaForCausalLM.from_pretrained(model_path).half().to(device)
    elif model_name == "opt":
        model_path = "facebook/opt-6.7b"
        model = OPTForCausalLM.from_pretrained(model_path).half().to(device)
    elif model_name == "bloomz":
        model_path = "bigscience/bloomz-7b1"
        model = BloomForCausalLM.from_pretrained(model_path).half().to(device)
    elif model_name == "rwkv":
        model_path = "RWKV/rwkv-raven-7b"
        model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def experiment(
    input_path, 
    output_path,
    model,
    tokenizer,
    batch_size=1,
    device="cuda",
    max_new_tokens=100,
    temperature=0.0,
    top_p=1.0,
):
    print(f"Reading {input_path}...")
    
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    prompts, examples, all_model_documents = prepare_prompts(input_path)
    
    responses = []
    print("Generating responses...")
    for batch_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts)/batch_size)):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # do_sample=temperature > 0.0,
            # temperature=temperature if temperature > 0 else None,
            # top_p=top_p if temperature > 0 else None,
            # return_dict_in_generate=False,
        )

        for i, generated_sequence in enumerate(outputs):
            input_ids = inputs["input_ids"][i]
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                )
            cleaned_text = text[prompt_length:]
            
            responses.append(cleaned_text)
    
    print(f"Writing to {output_path}...")
    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            f.write(json.dumps(output_example) + "\n")


if __name__ == '__main__':
    data_root = "/root/autodl-fs/mine/ushape/data/qa_data/"
    out_root = "/root/zhenting/ushape/qa_out"
    
    # for model_name in ["opt", "bloomz", "llama", "rwkv"]:
    for model_name in ["rwkv", ]:
        torch.cuda.empty_cache()
        model, tokenizer = prepare_model_and_tokenizer(model_name)
        # for num_doc in ["10", "20", "30"]:
        for num_doc in ["10", ]:    # 20 and 30 are too long
            data_dir_path = os.path.join(data_root, num_doc + "_total_documents")
            out_dir_path = os.path.join(out_root, num_doc + "_total_documents")
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)

            for gz_file in os.listdir(data_dir_path):
                input_path = os.path.join(data_dir_path, gz_file)
                s = gz_file.split(".")
                assert len(s) == 3
                output_path = os.path.join(out_dir_path, ".".join([s[0] + "-" + model_name, s[1], s[2]]))
                experiment(input_path, output_path, model, tokenizer)
        del model
        del tokenizer

    print("DONE!")