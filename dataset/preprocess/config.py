import os
import subprocess

def create_recipie(dataset_path, export_path, config_path, config_name, text_keys, num_process):
    recipe = f"""
    # Global Parameters
    project_name: 'finance-dataset'
    dataset_path: {dataset_path}
    export_path: {export_path}

    # Number of parallel processes
    np: {num_process}

    # Declare ALL Text Fields Upfront
    text_keys: {text_keys}
    
    # Process Pipeline
    process: 
    # normalize
    - whitespace_normalization_mapper:     # normalize different kinds of whitespaces to English whitespace.
    - clean_email_mapper:                                     
    - clean_html_mapper:                                      
    - clean_ip_mapper:                                       
    - clean_links_mapper:                                     
    - clean_copyright_mapper: 

    # filter                                
    - language_id_score_filter:             # filter text in specific language with language scores larger than a specific max value
        lang: 'en'
        min_score: 0.8                      # keep text in what language
    - maximum_line_length_filter:           # filter text with the maximum length of lines out of specific range
        min_len: 5
    - text_length_filter:                   # filter text with length out of specific range
        min_len: 5   
    - alphanumeric_filter:                  # filter text with alphabet/numeric ratio out of specific range.                  
        tokenization: false                                     
        min_ratio: 0.0                                          
        max_ratio: 0.9 
    - character_repetition_filter:          # filter text with the character repetition ratio out of specific range
        rep_len: 10                         # repetition length for char-level n-gram
        max_ratio: 0.6
    - special_characters_filter:            # filter text with special-char ratio out of specific range
        min_ratio: 0.0    
        max_ratio: 0.25
    - token_num_filter:                     # filter text with total token number out of specific range
        hf_tokenizer: 'Qwen/Qwen3-0.6B'
        min_num: 1   
        max_num: 10000
    
    # remove duplicates
    - document_deduplicator:                # deduplicate text samples using md5 hashing exact matching method
        lowercase: true
        ignore_non_character: false
    
    - document_simhash_deduplicator:        
        tokenization: space
        window_size: 6                     
        num_blocks: 6
        hamming_distance: 4
        lowercase: true
        ignore_pattern: null   
    """
    
    os.makedirs(config_path, exist_ok=True)
    print("path",os.path.join(config_path,config_name))
    with open(os.path.join(config_path,config_name), 'w') as f:
        f.write(recipe)
    
    print("recipe created")
    return


def statistical_analysis(stats_file):
    if os.path.exists(stats_file):
        overall_stats = pd.read_csv(stats_file)
        print("Overall Statistics:")
        print(overall_stats)
        
        # Extract key statistics for parameter tuning
        print("\n=== Analysis-Based Parameter Recommendations ===\n")
        
        # 1. Language Score Analysis - use 25th percentile
        lang_score_25th = overall_stats.loc[overall_stats['Unnamed: 0'] == '25%', 'lang_score'].values[0]
        lang_score_mean = overall_stats.loc[overall_stats['Unnamed: 0'] == 'mean', 'lang_score'].values[0]
        print(f"1. Language Score Filter:")
        print(f"   - Mean score: {lang_score_mean:.4f}")
        print(f"   - 25th percentile: {lang_score_25th:.4f}")
        print(f"   - Recommendation: Set min_score to {lang_score_25th:.4f} (filters bottom 25%)")
        
        # 2. Text Length Analysis - use 25th percentile
        text_len_25th = overall_stats.loc[overall_stats['Unnamed: 0'] == '25%', 'text_len'].values[0]
        text_len_mean = overall_stats.loc[overall_stats['Unnamed: 0'] == 'mean', 'text_len'].values[0]
        text_len_75th = overall_stats.loc[overall_stats['Unnamed: 0'] == '75%', 'text_len'].values[0]
        text_len_max = overall_stats.loc[overall_stats['Unnamed: 0'] == 'max', 'text_len'].values[0]
        print(f"\n2. Text Length Filter:")
        print(f"   - Mean length: {text_len_mean:.1f}")
        print(f"   - 25th percentile: {text_len_25th:.0f}")
        print(f"   - 75th percentile: {text_len_75th:.0f}")
        print(f"   - Recommendation: Set min_len={text_len_25th:.0f} (filters bottom 25%), max_len={text_len_75th * 1.5:.0f}")
        
        # 3. Alphanumeric Ratio Analysis - use 25th percentile
        alnum_25th = overall_stats.loc[overall_stats['Unnamed: 0'] == '25%', 'alnum_ratio'].values[0]
        alnum_mean = overall_stats.loc[overall_stats['Unnamed: 0'] == 'mean', 'alnum_ratio'].values[0]
        print(f"\n3. Alphanumeric Filter:")
        print(f"   - Mean ratio: {alnum_mean:.4f}")
        print(f"   - 25th percentile: {alnum_25th:.4f}")
        print(f"   - Recommendation: Set min_ratio to {alnum_25th:.4f} (filters bottom 25%)")
        
        # Generate optimized config based on 25th percentile
        optimized_config = {
            'language_id_score_filter': {
                'min_score': float(f"{lang_score_25th:.4f}")
            },
            'text_length_filter': {
                'min_len': int(text_len_25th),
                'max_len': int(text_len_75th * 1.5)  # Allow some flexibility above 75th percentile
            },
            'alphanumeric_filter': {
                'min_ratio': float(f"{alnum_25th:.4f}")
            }
        }
        
        print("\n=== Optimized Configuration (Bottom 25% Filtered) ===")
        import json
        print(json.dumps(optimized_config, indent=2))
        print("\n📊 Strategy: Using 25th percentile ensures only top 75% quality samples are retained")
    else:
        print("Statistics file not found. Analysis may still be running.")

    return 


if __name__=="__main__":
    dataset_path='../data_raw/Sujet-Finance-Instruct-177k'
    export_path='../data_processed/Sujet-Finance-Instruct-177k'
    data_name='qa_conversation.jsonl'
    
    config_path='../configs/Sujet-Finance-Instruct-177k'
    config_name='qa_conversation.yaml'
    
    text_keys=['answer', 'system_prompt', 'user_prompt']

    num_process=8
    create_recipie(os.path.join(dataset_path,data_name), os.path.join(export_path,data_name), config_path, config_name, text_keys, num_process)
    
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