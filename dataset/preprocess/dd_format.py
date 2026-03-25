import re
import os 
from datasets import load_dataset, DatasetDict
from utils import format_to_messages, content_format_to_messages,format_to_generalqa_eval

def main(load_path,train_save_path,eval_save_path,file_name,num_eval,system,user,assistant):
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    print(ds[1])
    ## shuffle dataset
    ds_shuffled = ds.shuffle(seed=42)
    
    num_ds=len(ds_shuffled)
    num_eval=200
    num_train = num_ds - num_eval
    
    train_ds = ds_shuffled.select(range(num_train))
    eval_ds = ds_shuffled.select(range(num_eval, len(ds_shuffled)))
    

    # map function with arguments 
    train_ds = train_ds.map(content_format_to_messages, 
                            fn_kwargs={
                                "system": system, 
                                "user": user,
                                "assistant": assistant
                            }, 
                            remove_columns=['instruction','context','response','category']
                        )
    # print(train_ds[1])
    train_ds.to_json(os.path.join(train_save_path, file_name))

    
    # map function with arguments 
    eval_ds = eval_ds.map(format_to_generalqa_eval, 
                            fn_kwargs={
                                "system": system, 
                                "user": user,
                                "assistant": assistant
                            }, 
                            remove_columns=['instruction','context','response','category']
                        )
    # print(train_ds[1])
    eval_ds.to_json(os.path.join(eval_save_path, file_name))



if __name__=="__main__":
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_raw/DatabricksDolly"
    train_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/DatabricksDolly/train"
    eval_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/DatabricksDolly/eval"
    file_name="summarization.jsonl"
    num_eval=100
    system="system"
    user="user"
    assistant="assistant"
    main(load_path,train_save_path,eval_save_path,file_name,num_eval,system,user,assistant)