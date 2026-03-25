import re
import os 

from datasets import load_dataset, DatasetDict

def format_to_multi_choice_eval():
    pass

if __name__=="__main__":
    # save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/Sujet-Finance-Instruct-177k"
    # load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_raw/Sujet-Finance-Instruct-177k"
    # file_name="yes_no_question.jsonl"
    
    # ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    # ds = ds.map(format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
    # print(ds[0])
    # ds.to_json(os.path.join(save_path, file_name))


    save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/Sujet-Finance-Instruct-177k"
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_raw/Sujet-Finance-Instruct-177k"
    file_name="qa_conversation.jsonl"
    
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    ds = ds.map(multi_turn_format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
    print(ds[0])
    ds.to_json(os.path.join(save_path, file_name))
