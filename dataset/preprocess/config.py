import os
import subprocess

def create_recipie(dataset_path, export_path, config_path, config_name, text_keys):
    recipe = f"""
    # Global Parameters
    project_name: 'finance-dataset'
    dataset_path: {dataset_path}
    export_path: {export_path}

    # Number of parallel processes
    np: 2

    # Declare ALL Text Fields Upfront
    text_keys: {text_keys}
    
    # Process Pipeline
    process: 
    # normalize
    - whitespace_normalization_mapper: 
    - clean_email_mapper:                                     
    - clean_html_mapper:                                      
    - clean_ip_mapper:                                       
    - clean_links_mapper:                                     
    - clean_copyright_mapper: 

    # filter  
    - language_id_score_filter:
        lang: 'en'
        min_score: 0.8
    - maximum_line_length_filter: # 104575811
        min_len: 5
    - text_length_filter: # 104573711
        min_len: 5
    - alphanumeric_filter:                                    
        tokenization: false                                     
        min_ratio: 0.0                                          
        max_ratio: 0.9 
    - flagged_words_filter: # 104576967
        lang: en
        max_ratio: 0.017
    - character_repetition_filter: # 104630030
        rep_len: 10
        max_ratio: 0.6
    - special_characters_filter: 
        min_ratio: 0.0    
        max_ratio: 0.25

    - token_num_filter: 
        hf_tokenizer: 'Qwen/Qwen3-0.6B'
        min_num: 1   
        max_num: 10000

    # remove duplicates
    - document_deduplicator:
        lowercase: true
    - document_simhash_deduplicator:  
        tokenization: space
        window_size: 3
        lowercase: true
        num_blocks: 9
        hamming_distance: 7
    """

    os.makedirs(config_path, exist_ok=True)
    print("path",os.path.join(config_path,config_name))
    with open(os.path.join(config_path,config_name), 'w') as f:
        f.write(recipe)
    
    print("recipe created")
    return


if __name__=="__main__":
    dataset_path='../data_raw/FinanceReasoningSynthetic'
    export_path='../data_processed/FinanceReasoningSynthetic'
    data_name='financereasoningsynthetic.jsonl'
    
    config_path='../configs/FinanceReasoningSynthetic'
    config_name='financereasoningsynthetic.yaml'
    
    text_keys=["question","cot","answer"]
    create_recipie(os.path.join(dataset_path,data_name), os.path.join(export_path,data_name), config_path, config_name, text_keys)
    
    command = ["dj-process", "--config", os.path.join(config_path,config_name)]
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, # merge error logs into standard output
        text=True
    )
    
    # print the output line by line as it is generated
    for line in process.stdout:
        print(f"[DJ-LOG]: {line.strip()}")

    process.wait() # ensure the process is fully finished