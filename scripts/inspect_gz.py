from xopen import xopen
import json


with xopen("/root/zhenting/ushape/data/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_2.jsonl.gz") as fin:
    a = 0
    for line in fin:
        # print(line)
        d = json.loads(line)
        print(d['answers'])
        import pdb; pdb.set_trace()
        a += 1
    print()
    
print("DONE")