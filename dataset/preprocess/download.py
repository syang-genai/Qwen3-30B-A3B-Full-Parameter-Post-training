import os 
from datasets import load_dataset, DatasetDict

def SujetFinanceInstruct(save_path):
    ds = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")
    ds = ds["train"]
    ds = ds.remove_columns(['Unnamed: 0', 'inputs','dataset', 'index_level', 'conversation_id'])

    target_labels=ds.unique('task_type')
    ds_split = {label: ds.filter(lambda example: example["task_type"] == label) for label in target_labels}
    ds_split= DatasetDict(ds_split)
    
    print("sujetfinanceinstruct category", target_labels)
    for key in ds_split.keys():
        print("category type: ",key,"category exmaple: ", ds_split[key][0])
        ds_split[key].to_json(os.path.join(save_path,key+".jsonl"))
    return

def FinancialInstructionAq22(save_path):
    ds = load_dataset("DeividasM/financial-instruction-aq22")
    ds = ds["train"]
    ds = ds.select(range(int(len(ds)*0.1)))
    ds.to_json(os.path.join(save_path, "knowledge_qa.jsonl"))
    print("financialinstructionaq22 exmaple: ", ds[0])
    return 

# def MathInstruct(save_path):
#     ds = load_dataset("TIGER-Lab/MathInstruct")
#     ds = ds["train"]
#     data_source=ds.unique('source')
#     print(data_source)
    
#     ds_split = {source: ds.filter(lambda example: example["source"] == source) for source in data_source}
#     ds_split= DatasetDict(ds_split)
    
#     for key in ds_split.keys():
#         if key=="data/PoT/numglue.json" or key=="data/CoT/numglue.json":
#             continue
        
#         print(ds_split[key][0])
#         ds_split[key].to_json(os.path.join(save_path,key.replace("josn","jsonl")))
#     return


def DatabricksDolly(save_path):
    ds = load_dataset("databricks/databricks-dolly-15k")
    ds = ds["train"]
    data_cat=ds.unique('category')
    
    ds_split = {cat: ds.filter(lambda example: example["category"] == cat) for cat in data_cat}
    ds_split= DatasetDict(ds_split)
    
    print("databricksdolly category", data_cat)
    for key in ds_split.keys():
        if key=="creative_writing":
            continue
        print("category type: ",key,"category exmaple: ", ds_split[key][0])
        ds_split[key].to_json(os.path.join(save_path,key+".jsonl"))
    return


def FinanceInstruct500k(save_path):
    ds = load_dataset("Josephgflowers/Finance-Instruct-500k")
    ds = ds["train"]
    ds = ds.remove_columns(["system"])
    ds = ds.select(range(int(len(ds)*0.1)))
    ds.to_json(os.path.join(save_path, "financeinstruct500k.jsonl"))
    print("financeinstruct500k exmaple: ", ds[0])
    return


def FinanceReasoningSynthetic(save_path):
    ds = load_dataset("RinKana/finance-reasoning-synthetic")
    ds = ds["train"]
    ds.to_json(os.path.join(save_path, "financereasoningsynthetic.jsonl"))
    print("financereasoningsynthetic: ", ds[0])
    return


if __name__=="__main__":
    # SujetFinanceInstruct(save_path="../data_raw/Sujet-Finance-Instruct-177k")
    FinancialInstructionAq22(save_path="../data_raw/Financial-Instruction-AQ22")
    # MathInstruct(save_path="../data_raw/MathInstruct")
    # DatabricksDolly(save_path="../data_raw/DatabricksDolly")
    # FinanceInstruct500k(save_path="../data_raw/FinanceInstruct500k")
    # FinanceReasoningSynthetic(save_path="../data_raw/FinanceReasoningSynthetic")