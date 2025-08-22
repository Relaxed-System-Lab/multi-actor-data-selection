'''
This code is adopted from https://github.com/princeton-nlp/LESS.
'''

from difflib import SequenceMatcher
# from util import *
from typing import  Sequence
from collections.abc import Iterable
import torch


def encode_with_content_format(example, tokenizer, max_seq_length=1024):
    tokenizer_chunk = int(max_seq_length)
    overlap_length = int(max_seq_length/4)
    content = example['content']
    start = 0
    ids = tokenizer(content, return_tensors='pt', truncation=False, add_special_tokens=False).input_ids.flatten()
    #ids = tokenizer(content + tokenizer.eos_token, return_tensors='pt', truncation=False).input_ids.flatten()
    token_length = 1024
    while start < len(ids):
        input_ids = ids[start : min(start + tokenizer_chunk, len(ids))]
        labels = input_ids.clone()
        # print("**********************************************************")
        # print("input_ids")
        # print(tokenizer.decode(input_ids.flatten()))
        # print(len(input_ids.flatten()))
        # print("**********************************************************")

        yield {'input_ids':input_ids, 'labels':labels, "ids":example["ids"]}
        # next start
        if start + tokenizer_chunk >= token_length:
            break
        else:
            start = start + tokenizer_chunk - overlap_length




def encode_with_question_answer_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['question'].endswith((' ', '\n', '\t')) and not example['answer'].startswith((' ', '\n', '\t')):
        example_text = example['question'] + ' ' + example['answer']
    else:
        example_text = example['question'] + example['answer']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['question'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # print(tokenizer.decode(labels[:, :tokenized_prompt.input_ids.shape[1]].flatten()))
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    # print("##########################################################")
    # print("piece")
    # print(example_text)
    # print("**********************************************************")
    # print("input_ids")
    # print(tokenizer.decode(input_ids.flatten()))
    # print("**********************************************************")
    # print("labels")
    # print(tokenizer.decode(labels.flatten()))
    # print("##########################################################")
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        "ids":example["ids"],
        'attention_mask': attention_mask.flatten(),
    }

def flatten(xs: Sequence) -> list:
    """Flatten a nested list."""

    def _flatten(ys):
        for y in ys:
            if isinstance(y, Iterable) and not isinstance(y, (str, bytes)):
                yield from _flatten(y)
            else:
                yield y

    return list(_flatten(xs))
