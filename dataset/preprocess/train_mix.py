import os 
from datasets import load_dataset, concatenate_datasets

def main(files_counts,save_path):
    ds_list=[]
    for f, c in files_counts.items():
        ds=load_dataset("json", data_files=f,split="train")
        ds_list.append(ds.select(range(c)))
    
    dataset=concatenate_datasets(ds_list)
    dataset = dataset.shuffle(seed=42)
    
    dataset.save_to_disk(save_path)
    return 


if __name__=="__main__":
    files_counts=dict()
    base_path=[
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly",
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/FinanceInstruct500k",
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/Financial-Instruction-AQ22",
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/Sujet-Finance-Instruct-177k",
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/FinanceReasoningSynthetic",
        ]
    for base in base_path:
        for filename in os.listdir(base):
            full_path = os.path.join(base, filename)
            files_counts[full_path]=100

    save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/sft_train_mix"
    main(files_counts,save_path)
