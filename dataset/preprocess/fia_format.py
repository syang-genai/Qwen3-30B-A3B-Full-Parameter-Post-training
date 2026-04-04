import re
import os 
from datasets import load_dataset, DatasetDict
from utils import format_to_messages, nosystem_format_to_generalqa_eval


def main(load_path,train_save_path,eval_save_path,file_name,train_eval_ratio,user,assistant):
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    print(ds[1])
    # shuffle dataset
    ds_shuffled = ds.shuffle(seed=42)
    
    total_num=len(ds)
    print("total_num",total_num)
    num_eval=int(total_num*train_eval_ratio)
    num_train=total_num-num_eval
    
    train_ds = ds_shuffled.select(range(num_train))
    eval_ds = ds_shuffled.select(range(num_eval, len(ds_shuffled)))
    
    # map function with arguments 
    train_ds = train_ds.map(format_to_messages, 
                            fn_kwargs={
                                "user": user,
                                "assistant": assistant
                            }, 
                            remove_columns=[user,assistant]
                        )
    print("train example: ", train_ds[1])
    train_ds.to_json(os.path.join(train_save_path, file_name))
    
    # map function with arguments 
    eval_ds = eval_ds.map(nosystem_format_to_generalqa_eval, 
                            fn_kwargs={
                                "user": user,
                                "assistant": assistant
                            }, 
                            remove_columns=[user,assistant]
                        )
    print("eval example: ",eval_ds[1])
    eval_ds.to_json(os.path.join(eval_save_path, file_name))
    return 


if __name__=="__main__":
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/Financial-Instruction-AQ22"
    train_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_dataset/Financial-Instruction-AQ22"
    eval_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/eval_dataset/Financial-Instruction-AQ22"
    file_name="knowledge_qa.jsonl"
    
    cot="cot"
    user="instruction"
    assistant="output"
    train_eval_ratio=0.2
    main(load_path,train_save_path,eval_save_path,file_name,train_eval_ratio,user,assistant)