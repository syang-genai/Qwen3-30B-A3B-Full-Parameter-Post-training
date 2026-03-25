import re
import os 

from datasets import load_dataset, DatasetDict

# define think function
def format_to_messages_think(example):
    return {
        "messages": [
            {"role": "user", "content": example['instruction']+"/no_think"},
            {"role": "assistant", "content": example['cot']+ example['response']}
        ]
    }


if __name__=="__main__":
    save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/FinanceReasoningSynthetic"
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_raw/FinanceReasoningSynthetic"
    
    file_name="financereasoningsynthetic.jsonl"
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    print(ds[1])
    
    ds = ds.map(content_format_to_messages, remove_columns=['instruction','context','response','category'])
    # print(ds[1])
    # ds.to_json(os.path.join(save_path, file_name))
