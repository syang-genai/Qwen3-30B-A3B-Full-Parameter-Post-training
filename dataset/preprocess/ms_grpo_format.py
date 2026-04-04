import os 
from datasets import load_dataset

def grpo_nothink_single_turn_chat(example):
    grpo_example={"messages":list()}
    for item in example["messages"]:
        if item["role"]=="system":
            grpo_example["messages"].append({"role":"system","content":item["content"]})
        elif item["role"]=="user":
            item["content"]=item["content"]+" "+"/no_think"
            grpo_example["messages"].append({"role":"user","content":item["content"]})
        elif item["role"]=="assistant":
            item["content"]="<think>\n\n</think>\n\n"+item["content"]
            grpo_example["response"]=item["content"]
    return grpo_example


def grpo_think_single_turn_chat(example):
    grpo_example={"messages":list()}
    for item in example["messages"]:
        if item["role"]=="system":
            grpo_example["messages"].append({"role":"system","content":item["content"]})
        elif item["role"]=="user":
            grpo_example["messages"].append({"role":"user","content":item["content"]})
        elif item["role"]=="assistant":
            grpo_example["response"]=item["content"]
    return grpo_example

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
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_dataset/Sujet-Finance-Instruct-177k"
    save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/msswift_grpo_train_dataset/Sujet-Finance-Instruct-177k"    
    files=["qa_conversation.jsonl"]
    
    map_func=multi_turn_chat
    main(load_path,save_path,files, map_func)
    