import re
import os 
from datasets import load_dataset, DatasetDict


def think_format_to_messages(example,cot,user,assistant):
    return {
        "messages": [
            {"role": "user", "content": example[user]+"/think"},
            {"role": "assistant", "content": example[cot]+example[assistant]}
        ]
    }


def nosystem_think_format_to_generalqa_eval(example,cot,user,assistant):
    return {
        "messages": [
            {"role": "user", "content": example[user]+"/think"}],
        "response": example[cot]+example[assistant]
    }

def main(load_path,train_save_path,eval_save_path,file_name,train_eval_ratio,cot,user,assistant):
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    ds_shuffled = ds.shuffle(seed=42)
    print("example",ds[0])
    
    total_num=len(ds)
    print("total_num",total_num)
    num_eval=int(total_num*train_eval_ratio)
    num_train=total_num-num_eval
    
    train_ds = ds_shuffled.select(range(num_train))
    eval_ds = ds_shuffled.select(range(num_eval, len(ds_shuffled)))
    
    # map function with arguments 
    train_ds = train_ds.map(think_format_to_messages, 
                            fn_kwargs={
                                "cot": cot, 
                                "user": user,
                                "assistant": assistant
                            }, 
                            remove_columns=[cot, user, assistant]
                        )
    print("train example: ", train_ds[1])
    train_ds.to_json(os.path.join(train_save_path, file_name))

    
    # map function with arguments 
    eval_ds = eval_ds.map(nosystem_think_format_to_generalqa_eval, 
                            fn_kwargs={
                                "cot": cot, 
                                "user": user,
                                "assistant": assistant
                            }, 
                            remove_columns=[cot, user, assistant]
                        )
    
    print("eval example: ",eval_ds[1])
    eval_ds.to_json(os.path.join(eval_save_path, file_name))
    return

if __name__=="__main__":
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/FinanceReasoningSynthetic"
    train_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/FinanceReasoningSynthetic"
    eval_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/eval_data_format/FinanceReasoningSynthetic"
    file_name="financereasoningsynthetic.jsonl"
    
    user="question"
    assistant="answer"
    cot="cot"
    train_eval_ratio=0.2
    main(load_path,train_save_path,eval_save_path,file_name,train_eval_ratio,cot,user,assistant)