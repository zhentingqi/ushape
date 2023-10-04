import pdb
import math
import torch
from transformers import AutoTokenizer
from transformers import OPTForCausalLM, BloomForCausalLM, LlamaForCausalLM
from tqdm import tqdm
from xopen import xopen
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from copy import deepcopy


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


def prepare_prompts(data_path):
    prompts, examples, all_model_documents = [], [], []
    print("Reading data file...")
    with xopen(data_path) as fin:
        for line in fin:
            input_example = json.loads(line)

            question = input_example["question"]
            documents = [Document.from_dict(ctx) for ctx in deepcopy(input_example["ctxs"])]
            prompt = get_qa_prompt(question, documents, mention_random_ordering=False, query_aware_contextualization=False)
                    
        prompts.append(prompt)
        examples.append(deepcopy(input_example))
        all_model_documents.append(documents)

    return prompts, examples, all_model_documents


def prepare_model_and_tokenizer(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    if "opt" in model_path:
        model = OPTForCausalLM.from_pretrained(model_path).to(device)
    elif "bloomz" in model_path:
        model = BloomForCausalLM.from_pretrained(model_path).to(device)
    elif "llama" in model_path:
        model = LlamaForCausalLM.from_pretrained(model_path).to(device)

    return model, tokenizer


def experiment(
    data_path, 
    output_path,
    model_path,
    batch_size=1,
    device="cuda",
    max_new_tokens=100,
    temperature=0.0,
    top_p=1.0,
):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    prompts, examples, all_model_documents = prepare_prompts(data_path)
    model, tokenizer = prepare_model_and_tokenizer(model_path)
    
    responses = []
    print("Generating responses...")
    for batch_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts)/batch_size)):
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            return_dict_in_generate=False,
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
            
            pdb.set_trace()

    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_path
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            f.write(json.dumps(output_example) + "\n")


if __name__ == '__main__':
    data_path = "/root/zhenting/ushape/data/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl.gz"
    output_path = "/root/zhenting/ushape/data/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4-responses.jsonl.gz"
    model_path = "/root/autodl-fs/llama-to-hf"
    experiment(data_path, output_path, model_path)
