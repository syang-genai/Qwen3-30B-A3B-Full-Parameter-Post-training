import os 
from datasets import load_dataset

def nothink_single_turn_chat(example):
    # nothink_example={"messages":list()}
    for item in example["messages"]:
        if item["role"]=="user":
            item["content"]=item["content"]+" "+"/no_think"
        elif item["role"]=="assistant":
            item["content"]="<think>\n\n</think>\n\n"+item["content"]
    return example


def multi_turn_chat(example):
    for item in example["messages"][:-1]:
        if item["role"]=="user":
            item["content"]=item["content"]+" "+"/no_think"
        elif item["role"]=="assistant":
            item["content"]="<think>\n\n</think>\n\n"+item["content"]
            item["loss"] = False
    
    item=example["messages"][-1]
    item["content"]="<think>\n\n</think>\n\n"+item["content"]
    item["loss"] = True
    return example


def main(load_path,save_path,files, map_func):
    for file in files:
        load_file=os.path.join(load_path,file)
        print("load_file", load_file)
        
        ds = load_dataset("json", data_files=load_file, split="train")
        print("dataset before format", ds[0])
        ds = ds.map(map_func)
        
        print("dataset after format", ds[0])
        ds.to_json(os.path.join(save_path, "financereasoningsynthetic.jsonl"))
    return


if __name__=="__main__":
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/Sujet-Finance-Instruct-177k"
    save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format_msswift/Sujet-Finance-Instruct-177k"    
    files=["qa_conversation.jsonl"]
    
    map_func=multi_turn_chat
    main(load_path,save_path,files, map_func)
    