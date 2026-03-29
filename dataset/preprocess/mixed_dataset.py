from datasets import load_dataset, concatenate_datasets

def main(files_counts,save_path):
    ds_list=[]
    for f, c in files_counts.items():
        ds=load_dataset("json", data_files=f,split="train")
        ds_list.append(ds)
    
    dataset=concatenate_datasets(ds_list)
    dataset = dataset.shuffle(seed=42)
    
    dataset.save_to_disk(save_path)
    return 


if __name__=="__main__":
    files_counts={
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly/brainstorming.jsonl":1000, 
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly/classification.jsonl":1000,
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly/closed_qa.jsonl":1000,
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly/general_qa.jsonl":1000,
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly/information_extraction.jsonl":1000,
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly/open_qa.jsonl":1000,
        "/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/DatabricksDolly/summarization.jsonl":1000
    }
    
    save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/sft_train_mix"
    main(files_counts,save_path)