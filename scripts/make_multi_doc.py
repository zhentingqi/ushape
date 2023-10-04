import pdb
import json
from xopen import xopen


src_gz = "/root/zhenting/ushape/data/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz"

with xopen(src_gz) as fin:
    for gold_pos in [1, 2, 3, 5, 6, 7, 8]:
        tgt_gz = f"/root/zhenting/ushape/data/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_{gold_pos}.jsonl.gz"
        with xopen(tgt_gz, "w") as fout:                
            for line in fin:
                qa_retrieval_result = json.loads(line)
                gold_ctx = qa_retrieval_result['ctxs'][0]
                assert gold_ctx['isgold'] is True
                qa_retrieval_result['ctxs'].pop(0)
                qa_retrieval_result['ctxs'].insert(gold_pos, gold_ctx)
                fout.write(json.dumps(qa_retrieval_result) + "\n")