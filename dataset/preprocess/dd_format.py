import re
import os 

from datasets import load_dataset, DatasetDict


def format_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['response']}
        ]
    }


def content_format_to_messages(example):
    return {
        "messages": [
            {"role": "system", "content": example['context']},
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['response']}
        ]
    }


def multi_turn_format_to_messages(example):
    def universal_chat_splitter(text):
        """
        Splits a raw string into a structured list of messages (role/content).
        Handles explicit labels (User:/Assistant:) and implicit 'floating' 
        sentences separated by triple newlines (\n\n\n).
        """
        
        # 1. Define split points: Explicit labels or 3+ consecutive newlines
        # Using a capturing group (...) keeps the delimiters in the resulting list
        split_pattern = r"((?:User|Assistant):|\n{3,})"
        parts = re.split(split_pattern, text)
        
        history = []
        # Default starting role (usually 'user' for chat datasets)
        current_role = "user" 
        
        # 2. Iterate through the split parts
        for i in range(len(parts)):
            segment = parts[i].strip()
            
            # Skip empty strings or segments that are just whitespace/newlines
            if not segment or re.match(r"^\n+$", segment):
                continue
                
            # 3. Role detection logic
            if segment == "User:":
                current_role = "user"
                continue
            elif segment == "Assistant:":
                current_role = "assistant"
                continue
            # Handle cases where the label and content are in the same segment
            elif segment.startswith("User:"):
                current_role = "user"
                segment = segment.replace("User:", "").strip()
            elif segment.startswith("Assistant:"):
                current_role = "assistant"
                segment = segment.replace("Assistant:", "").strip()
            
            # 4. Content Processing
            if segment:
                # Merge content if the detected role is the same as the last message
                # (Prevents broken 'user-user' sequences which some trainers dislike)
                if history and history[-1]["role"] == current_role:
                    history[-1]["content"] += "\n\n" + segment
                else:
                    history.append({
                        "role": current_role,
                        "content": segment
                    })
                
                # 5. Automatic Role Toggle
                # If the next segment is a 'floating' sentence (no label), 
                # it will correctly assume the opposite role to maintain the flow.
                current_role = "assistant" if current_role == "user" else "user"

        return history
    return {"messages":[{"role": "system", "content": example["system_prompt"]}]+universal_chat_splitter(example['user_prompt'])+[{"role": "assistant", "content": example["answer"]}]}

if __name__=="__main__":
    save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/DatabricksDolly"
    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_raw/DatabricksDolly"
    
    file_name="summarization.jsonl"
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    print(ds[1])
    
    ds = ds.map(content_format_to_messages, remove_columns=['instruction','context','response','category'])
    print(ds[1])
    ds.to_json(os.path.join(save_path, file_name))