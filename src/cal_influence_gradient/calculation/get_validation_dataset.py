'''
This code is adopted from https://github.com/princeton-nlp/LESS.
'''
import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
import datasets

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def get_bbh_dataset(data_dir: str,
                    tokenizer: PreTrainedTokenizerBase,
                    max_length: int,
                    use_chat_format: bool = True,
                    chat_format: str = "tulu",
                    **kwargs):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows: 

    Query: 
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    file = f"{data_dir}/eval/bbh/bbh-three-shot.json"

    bbh_few_shot_examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    # there are multiple tasks in the bbh dataset
    # each task has 3 examples
    for task in bbh_few_shot_examples:
        few_shot_exs = bbh_few_shot_examples[task]

        stuff = few_shot_exs.split("\n\n")
        exes = stuff[-3:]
        task_prompt = "\n\n".join(stuff[:-3])

        def form_icl(exs):
            string = ""
            for ex in exs:
                question, answer = ex.split("\nA:")
                string += question + "\nA:" + answer
                string += "\n\n"
            return string

        for i in range(len(exes)):
            target_ex = exes[i]
            other_exes = exes[:i] + exes[i+1:]
            icl = form_icl(other_exes)
            question, answer = target_ex.split("\nA:")

            if use_chat_format:
                if chat_format == "tulu":  # we follow the tulu instruction tuning format
                    question = "<|user|>\n" + task_prompt.strip() + "\n\n" + icl + \
                        f"{question}" + "\n<|assistant|>\nA:"
                else:
                    question = f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
            else:
                question = task_prompt.strip() + "\n\n" + \
                    f"{question}" + "\nA:"
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, question, answer, max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_tydiqa_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:  

    Query: 
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer: 

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
        "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
        "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
        "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
        "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
        "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。", "文章:", "问题:", "答案:")

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    file = os.path.join(f"{data_dir}/tydiqa/dev", file_name)

    examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, lang in enumerate(examples):
        example = examples[lang][0]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[lang]
        prompt += p_template + " " + \
            format(example["context"]) + "\n" + q_template + \
            " " + format(example["question"]) + "\n"
        answer = " " + format(example["answers"][0]["text"])
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_mmlu_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     **kwargs):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(train_df, subject, i=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        prompt += format_example(train_df, i, include_answer=False)
        return prompt

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        return prompt

    k = 10
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(mmlu_data_dir, "test", subject + "_test.csv"), header=None
        )[: k]
        for i in range(k):
            prompt = gen_prompt(dev_df, subject, i)
            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]

            if use_chat_format:
                if chat_format == "tulu":
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
                else:
                    # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                    prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
            else:
                prompt = prompt
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset

def get_arceasy_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "arc_easy")

    examples = datasets.load_dataset(data_dir)
    examples = examples["test"]

    for i, _ in enumerate(examples):
        choices = ""
        for j in range(len(examples["choices"][i]["text"])):
            choices += examples["choices"][i]["label"][j] + ": " + examples["choices"][i]["text"][j] + "\n"
        prompt = examples["question"][i] + "\n" + choices
        answer = examples["answerKey"][i]

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_arcchallenge_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "arc_challenge")

    examples = datasets.load_dataset(data_dir)
    examples = examples["test"]

    for i, _ in enumerate(examples):
        choices = ""
        for j in range(len(examples["choices"][i]["text"])):
            choices += examples["choices"][i]["label"][j] + ": " + examples["choices"][i]["text"][j] + "\n"
        prompt = examples["question"][i] + "\n" + choices
        answer = examples["answerKey"][i]

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_boolq_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "boolq")

    examples = datasets.load_dataset(data_dir)
    examples = examples["validation"]
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:")}    

    for i, _ in enumerate(examples):
        prompt, p_template, q_template, a_template = encoding_templates_with_context["english"]
        prompt += p_template + " " + \
            format(examples["passage"][i]) + "\n" + q_template + \
            " " + format(examples["question"][i]) + "\n"
        answer = " " + format(examples["answer"][i])

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_commonsenseqa_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "commonsense_qa")

    examples = datasets.load_dataset(data_dir)
    examples = examples["test"]

    for i, _ in enumerate(examples):
        choices = ""
        for j in range(len(examples["choices"][i]["text"])):
            choices += examples["choices"][i]["label"][j] + ": " + examples["choices"][i]["text"][j] + "\n"
        prompt = examples["question"][i] + "\n" + choices
        answer = examples["answerKey"][i]

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_openbookqa_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "openbookqa")

    examples = datasets.load_dataset(data_dir)
    examples = examples["test"]

    for i, _ in enumerate(examples):
        choices = ""
        for j in range(len(examples["choices"][i]["text"])):
            choices += examples["choices"][i]["label"][j] + ": " + examples["choices"][i]["text"][j] + "\n"
        prompt = examples["question_stem"][i] + "\n" + choices
        answer = examples["answerKey"][i]

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_race_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "race")

    examples = datasets.load_dataset(data_dir)
    examples = examples["test"]
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:")}    
   

    for i, _ in enumerate(examples):
        choices = ""
        for j in range(len(examples["options"][i])):
            choices += chr(ord('A') + j) + ": " + examples["options"][i][j] + "\n"
        prompt, p_template, q_template, a_template = encoding_templates_with_context["english"]
        prompt += p_template + " " + \
            format(examples["article"][i]) + "\n" + q_template + \
            " " + format(examples["question"][i]) + "\n" + \
            format(choices)
        answer = " " + format(examples["answer"][i])

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_sciq_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "sciq")

    examples = datasets.load_dataset(data_dir)
    examples = examples["test"]   

    for i, _ in enumerate(examples):
        prompt = examples["question"][i] + "\n"
        for j in range(1,5):
            if j < 4:
                prompt += chr(64 + j) + ": " + examples[f"distractor{j}"][i] + "\n"
            else: 
                prompt += chr(64 + j) + ": " + examples["correct_answer"][i] + "\n"
        
        answer = " " + format(examples["support"][i])

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_socialiqa_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "social_iqa")
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:")}
    examples = datasets.load_dataset(data_dir)  
    examples = examples["validation"]

    for i, _ in enumerate(examples):
        choices = ""
        for j in range(1,4):
            j = chr(64+j)
            choices += j + ": " + examples[f"answer{j}"][i] + "\n"

        prompt, p_template, q_template, a_template = encoding_templates_with_context["english"]
        prompt += p_template + " " + \
            format(examples["context"][i]) + "\n" + q_template + \
            " " + format(examples["question"][i]) + "\n" + \
            format(choices)
        answer = " " + format(chr(int(examples["label"][i]) + 64))

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_winogrande_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "winogrande")
    examples = datasets.load_dataset(data_dir)  
    examples = examples["validation"]

    for i, _ in enumerate(examples):
        choices = ""
        for j in range(1,3):
            choices += chr(64+j) + ": " + examples[f"option{j}"][i] + "\n"

        
        prompt = format(examples["sentence"][i]) + "\n" + "Option:" + \
            "\n" + format(choices)
        answer = " " + format(chr(int(examples["answer"][i]) + 64))

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_hellaswag_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "hellaswag")
    encoding_templates_with_context = {
        "english": ("Finish the following sentence based on the information.", "Sentence:", "Choices:", "Answer:")}
    examples = datasets.load_dataset(data_dir)  
    examples = examples["validation"]

    for i, _ in enumerate(examples):
        endings = ""
        for j in range(4):
            endings += chr(65 + j) + ": " + examples["endings"][i][j] + "\n"

        prompt, p_template, q_template, a_template = encoding_templates_with_context["english"]
        prompt += p_template + "\n" + \
            format(examples["ctx"][i]) + "____\n" + q_template + "\n" + \
            format(endings)
        answer = " " + format(chr(int(examples["label"][i]) + 65))

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_lambada_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "lambada")
    encoding_templates_with_context = {
        "english": ("Fill the last word based on the information.", "Sentence:", "Choices:", "Answer:")}
    examples = datasets.load_dataset(data_dir)  
    examples = examples["validation"]

    for i, _ in enumerate(examples):
        prompt, p_template, q_template, a_template = encoding_templates_with_context["english"]
        prompt += p_template + "\n" + \
            format(examples["until_last"][i]) + " _____\n"
        
        answer = " " + format(examples["last_word"][i])

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_jeopardy_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "jeopardy")
    encoding_templates_with_context = {
        "english": ("According to the description, give an one word answer.", "Description:", "Choices:", "Answer:")}
    examples = datasets.load_dataset(data_dir)  
    examples = examples["validation"]

    for i, _ in enumerate(examples):
        prompt, p_template, q_template, a_template = encoding_templates_with_context["english"]
        prompt += "\n" + p_template + "\n" + \
            format(examples["context"][i]) + "\n"
        
        answer = " " + format(examples["continuation"][i])

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset
        
def get_squad_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    data_dir = os.path.join(data_dir, "our_testset", "squad")
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:")}
    examples = datasets.load_dataset(data_dir)  
    examples = examples["validation"]

    for i, _ in enumerate(examples):
        if i > 500: break
        prompt, p_template, q_template, a_template = encoding_templates_with_context["english"]
        prompt += p_template + " " + format(examples["context"][i]) + "\n" + q_template + \
                " " + format(examples["question"][i]) + "\n"
        answer = " " + examples["answers"][i]["text"][0]

        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
            else:
                # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
        else:
            prompt = prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.shuffle()
    dataset = dataset.select(range(500))
    return dataset

def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "bbh":
        return get_bbh_dataset(**kwargs)
    elif task == "tydiqa":
        return get_tydiqa_dataset(**kwargs)
    elif task == "mmlu":
        return get_mmlu_dataset(**kwargs)
    elif task == "arc_easy":
        return get_arceasy_dataset(**kwargs)
    elif task == "arc_challenge":
        return get_arcchallenge_dataset(**kwargs)
    elif task == "boolq":
        return get_boolq_dataset(**kwargs)
    elif task == "commonsense_qa":
        return get_commonsenseqa_dataset(**kwargs)
    elif task == "openbookqa":
        return get_openbookqa_dataset(**kwargs)
    elif task == "race":
        return get_race_dataset(**kwargs)
    elif task == "sciq":
        return get_sciq_dataset(**kwargs)
    elif task == "social_iqa":
        return get_socialiqa_dataset(**kwargs)
    elif task == "winogrande":
        return get_winogrande_dataset(**kwargs)
    elif task == "hellaswag":
        return get_hellaswag_dataset(**kwargs)
    elif task == "lambada":
        return get_lambada_dataset(**kwargs)
    elif task == "jeopardy":
        return get_jeopardy_dataset(**kwargs)
    elif task == "squad":
        return get_squad_dataset(**kwargs)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
