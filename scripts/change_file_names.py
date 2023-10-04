import os


data_root = "/root/zhenting/ushape/data/qa_data"
for num_doc in ["10", "20", "30"]:
    dir_name = num_doc + "_total_documents"
    dir_path = os.path.join(data_root, dir_name)
    for gz_file in os.listdir(dir_path):
        old_path = data_root + dir_path + os.sep + gz_file
        assert "gold" in gz_file
        new_name = "gold" + gz_file.split("gold")[1]
        new_path = data_root + dir_path + os.sep + None