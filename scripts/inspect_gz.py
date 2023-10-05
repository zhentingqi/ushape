from xopen import xopen


with xopen("/root/zhenting/ushape/data/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_2.jsonl.gz") as fin:
    a = 0
    import pdb; pdb.set_trace()
    for line in fin:
        # print(line)
        a += 1
    import pdb; pdb.set_trace()
    print()
    
print("DONE")