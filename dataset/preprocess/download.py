# from huggingface_hub import login
import re
from datasets import load_dataset, DatasetDict

# login()
ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k", cache_dir="../data_raw")
#  print(ds)
ds = ds["train"]
ds = ds.remove_columns(['Unnamed: 0', 'inputs','dataset', 'index_level', 'conversation_id'])

target_labels=ds.unique('task_type')
# print("taget_labels", target_labels)

ds_split = {label: ds.filter(lambda example: example["task_type"] == label) for label in target_labels}
ds_split= DatasetDict(ds_split)
# print(ds_split['sentiment_analysis'][0])

# map and save
def format_to_messages(example):
    return {
        "messages": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["user_prompt"]},
            {"role": "assistant", "content": example["answer"]}
        ]
    }

ds_split['sentiment_analysis']=ds_split['sentiment_analysis'].map(format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
# print(ds_split['sentiment_analysis'][0])
# save for evaluation; check evaluation dataset formate!
ds_split['sentiment_analysis'].to_json("../data_processed/sentiment_analysis.jsonl")


# print(ds_split['qa'][0])
ds_split['qa']=ds_split['qa'].map(format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
# print(ds_split['qa'][0])
ds_split['qa'].to_json("../data_processed/qa.jsonl")

# print(ds_split['qa_with_context'][0])
ds_split['qa_with_context']=ds_split['qa_with_context'].map(format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
# print(ds_split['qa_with_context'][0])

# print(ds_split['yes_no_question'][0])
ds_split['yes_no_question']=ds_split['yes_no_question'].map(format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
# print(ds_split['yes_no_question'][0])

# print(ds_split['ner_sentiment_analysis'][0])
ds_split['ner_sentiment_analysis']=ds_split['ner_sentiment_analysis'].map(format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
# print(ds_split['ner_sentiment_analysis'][0])

# print(ds_split['topic_classification'][0])
ds_split['topic_classification']=ds_split['topic_classification'].map(format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
# print(ds_split['topic_classification'][0])


# multi_around conversation
# print(ds_split['qa_conversation'][3]) 

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

ds_split['qa_conversation']=ds_split['qa_conversation'].map(multi_turn_format_to_messages, remove_columns=["system_prompt","answer","user_prompt","task_type"])
print(ds_split['qa_conversation'][5])