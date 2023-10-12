import pdb
import os
import json
import matplotlib.pyplot as plt
from xopen import xopen
from collections import defaultdict


def plot():
    eval_root = "/root/zhenting/ushape/data/qa_eval"
    model_pe = {"opt": "Absolute Position Embedding", "llama": "Rotary", "bloomz": "ALiBi"}

    for num_doc in ["10", ]:
        model_performance = defaultdict(list)
        
        eval_dir_path = os.path.join(eval_root, num_doc + "_total_documents")
        for gz_file in os.listdir(eval_dir_path):
            gold_pos = int(gz_file[gz_file.find("at") + 3])
            model_name = gz_file[gz_file.find("at") + 5:gz_file.find("eval") - 1]
            
            score_list = []
            gz_file_path = os.path.join(eval_dir_path, gz_file)
            with xopen(gz_file_path) as fin:
                for line in fin:
                    example = json.loads(line)
                    score = example['metric_best_subspan_em']
                    score_list.append(score)

            avg_score = sum(score_list) / len(score_list)
            model_performance[model_name].append({"gold_pos":gold_pos, "avg_score":avg_score})
    
        plt.figure(1)
        plt.xlabel("position of gold document")
        plt.ylabel("score: best_subspan_em")
        pos_list = list(range(10))
        for model, performance in model_performance.items():
            assert len(performance) == int(num_doc)
            score_list = [0 for _ in range(10)]
            for d in performance:
                gold_pos, avg_score = d["gold_pos"], d["avg_score"]
                score_list[gold_pos] = avg_score
            
            plt.plot(pos_list, score_list, label=f"{model}: {model_pe[model]}")
        
        plt.legend()
        plt.savefig("./plot.png")
    

if __name__ == "__main__":
    plot()