import os 
from datasets import load_dataset, DatasetDict
from utils import format_to_messages, content_format_to_messages,format_to_generalqa_eval, multi_turn_format_to_messages


def main(load_path,train_save_path,eval_save_path,file_name,train_eval_ratio,system,user,assistant):
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    ds_shuffled = ds.shuffle(seed=42)
    
    total_num=len(ds)
    print("total_num",total_num)
    num_eval=int(total_num*train_eval_ratio)
    num_train=total_num-num_eval
    
    train_ds = ds_shuffled.select(range(num_train))
    eval_ds = ds_shuffled.select(range(num_eval, len(ds_shuffled)))
    
    # map function with arguments 
    train_ds = train_ds.map(content_format_to_messages, 
                            fn_kwargs={
                                "system":system, 
                                "user":user,
                                "assistant":assistant
                            }, 
                            remove_columns=[system,user,assistant,"task_type"]
                        )
    # print(train_ds[1])
    train_ds.to_json(os.path.join(train_save_path, file_name))
    print("train_ds example", train_ds[0])
    
    # map function with arguments 
    eval_ds = eval_ds.map(format_to_generalqa_eval, 
                            fn_kwargs={
                                "system":system, 
                                "user":user,
                                "assistant":assistant
                            }, 
                            remove_columns=[system,user,assistant,"task_type"]
                        )
    eval_ds.to_json(os.path.join(eval_save_path, file_name))
    print("eval_ds example", eval_ds[0])
    return

def main_multi_tune(load_path,train_save_path,file_name,system,user,assistant):
    ds = load_dataset("json", data_files=os.path.join(load_path,file_name),split="train")
    ds_shuffled = ds.shuffle(seed=42)
    
    total_num=len(ds)
    print("total_num",total_num)
    
    train_ds = ds_shuffled.select(range(total_num))
    
    # map function with arguments 
    train_ds = train_ds.map(multi_turn_format_to_messages, 
                            fn_kwargs={
                                "system":system, 
                                "user":user,
                                "assistant":assistant
                            }, 
                            remove_columns=[system,user,assistant,"task_type"]
                        )
    # print(train_ds[1])
    train_ds.to_json(os.path.join(train_save_path, file_name))
    print("train_ds example", train_ds[0])
    return 
    

if __name__=="__main__": 
    # load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/Sujet-Finance-Instruct-177k"
    # file_name="sentiment_analysis.jsonl"
    
    # train_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/Sujet-Finance-Instruct-177k"
    # eval_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/eval_data_format/Sujet-Finance-Instruct-177k"
    
    # system='system_prompt'
    # user='user_prompt'
    # assistant='answer'
    # train_eval_ratio=0.2
    # main(load_path,train_save_path,eval_save_path,file_name,train_eval_ratio,system,user,assistant)

    load_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/data_processed/Sujet-Finance-Instruct-177k"
    file_name="qa_conversation.jsonl"
    
    train_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/train_data_format/Sujet-Finance-Instruct-177k"
    eval_save_path="/root/Qwen3-30B-A3B-Full-Parameter-Post-training/dataset/eval_data_format/Sujet-Finance-Instruct-177k"
    
    system='system_prompt'
    user='user_prompt'
    assistant='answer'
    main_multi_tune(load_path,train_save_path,file_name,system,user,assistant)