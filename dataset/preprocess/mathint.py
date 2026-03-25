import re
import json

def process_qa_data(example, idx):
    instruction = example["instruction"]
    answer_text = example["answer"]

    question_match = re.search(r"(.*?)(?=\s*\(A\))", instruction, re.DOTALL)
    question = question_match.group(1).strip() if question_match else instruction

    options = {}
    for letter in ['A', 'B', 'C', 'D', 'E']:
        next_letters = "|".join([f"\\({l}\\)" for l in "ABCDE"[ "ABCDE".index(letter)+1 : ]])
        pattern = rf"\({letter}\)\s*(.*?)(?=\s*(?:{next_letters})|\s*Answer|$)"
        match = re.search(pattern, instruction, re.DOTALL)
        if match:
            options[letter] = match.group(1).strip()

    ans_match = re.search(r"The answer is ([A-E])", answer_text)
    final_answer = ans_match.group(1) if ans_match else ""

    return {
        "id": str(idx + 1),
        "question": question,
        **options, # 展开 A, B, C, D, E 到顶层
        "answer": final_answer
    }
