import contextlib
import argparse
import numpy as np
import torch
from datasets import Dataset

from less.data_selection.encode_dataset import (encode_with_content_format, encode_with_question_answer_format)

from less.data_selection.unify_data_format import (get_unify_dataset)
import gc
from transformers import (AutoTokenizer)
import datasets

import time
import multiprocessing
from tqdm import tqdm
import json

import logging
import jsonlines
import os
import pickle

logging.basicConfig(
    format='%(asctime)s %(filename)s:%(lineno)s [%(levelname)s] %(message)s', level=logging.INFO)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def read_list(fname):
    """从文件中读取保存的列表"""
    with open(fname, 'rb') as f:
        zc_list = pickle.load(f)
    return zc_list

        
def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--model_path', type=str,
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--output_path', type=str,
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--max_seq_length', type=int,
                           help='The path to the score file')
    argparser.add_argument('--sample_percentage', type=float,
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--processing_num_workers', type=int,
                           help='The name of the target task')
    argparser.add_argument('--data_seed', type=int,
                           help='The name of the target task')

    args = argparser.parse_args()

    return args

def get_training_dataset(all_file_paths, tokenizer=None, max_seq_length=1024, sample_percentage=1.0, processing_num_workers=1, seed=0, flag=True, slc=0):
    #为true时表示第一次进行tokenize
    lm_datasets = []
    # train_file_paths = load_train_files(all_file_paths)
    train_file_paths = [all_file_paths]
    print(train_file_paths)
    count = 0#计算ids
    for filepath in train_file_paths:
        # print(filepath)
        if flag==True:
            raw_datasets = load_raw_dataset(filepath, sample_percentage=sample_percentage, seed=seed,slc=slc)
            unified_datasets, count = get_unify_dataset(raw_datasets,count)
            # print(count)
        else:
            unified_datasets = 0
        print("data count",count)
        temp_datasets = encode_data(unified_datasets, tokenizer, max_seq_length, os.path.basename(filepath), processing_num_workers)
        print('temp_datasets len:',len(temp_datasets))
        global dataset_length
        dataset_length = len(temp_datasets)
        lm_datasets = lm_datasets + temp_datasets
    result_datasets = Dataset.from_list(lm_datasets)
    result_datasets.set_format(type="pt")
    print(result_datasets)
    return result_datasets


def load_raw_dataset(file_path, sample_size=None, sample_percentage=1.0, seed=0, slc=0):
    """ load raw dataset """
    processed_datasets = []
    if file_path.startswith("s3:"):
        #从s3读数据
        processed_datasets = read_s3_jsonl(file_path)
    else:
        #从本地读数据
        processed_datasets = []
        count = 0

        with jsonlines.open(file_path, 'r') as f:
            for line in f:
                count += 1
                if count >= (slc*1250) and count < (slc+1)*1250:
                    processed_datasets.append(line)
                # if len(processed_datasets) < 50:
                #     processed_datasets.append(line)
                # else: break
    processed_datasets = Dataset.from_list(processed_datasets)
    print(processed_datasets)
    
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]
    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset

def encode_data(raw_datasets, tokenizer, max_seq_length, filename, processing_num_workers=1):
    if processing_num_workers == 1:
        print("processing_num_workers == 1")
        return merge(raw_datasets, tokenizer, max_seq_length, filename, local_tokenized = False)
    else:
        print("processing_num_workers = {}".format(processing_num_workers))
        return multiprocess_merge(raw_datasets, tokenizer, max_seq_length, processing_num_workers)

def merge(raw_datasets, tokenizer, max_seq_length, fname, local_tokenized = False):
    #对文件内容进行chunk和tokenize
    # if already encoded, return
    zc_list = []
    start_time = time.time()
    fname = "/fs-computility/llm/shared/baitianyi/train/iclr/LESS/tokenize_record3/" + fname + ".txt"
    if local_tokenized==False:
        #本地无tokenize后的文件
        for example in raw_datasets:
            gen = encode_with_content_format(example, tokenizer, max_seq_length)#tokenize
            for result in gen:
                zc_list.append(result)
            # gen = encode_with_question_answer_format(example, tokenizer, max_seq_length)
            # zc_list.append(gen)
        # if not os.path.exists("../tokenize_record3/"):
        #     os.makedirs("../tokenize_record3/")
        # save_list(fname, zc_list)
    else:
        zc_list = read_list(fname)
    end_time = time.time()
    print("time is: ", end_time-start_time)
    return zc_list

def task_func(i, raw_datasets, tokenizer, max_seq_length, begin, end, share_var, share_lock):
    zc_list = []
    for example in tqdm(raw_datasets[begin:end], desc="Processing"):
        gen = encode_with_content_format(example, tokenizer, max_seq_length)
        for result in gen:
            zc_list.append(result)
        # gen = encode_with_question_answer_format(example, tokenizer, max_seq_length)
        # zc_list.append(gen)
    if not os.path.exists("/fs-computility/llm/shared/baitianyi/train/iclr/LESS/tmp"):
        os.makedirs("/fs-computility/llm/shared/baitianyi/train/iclr/LESS/tmp")
    dataset = Dataset.from_list(zc_list)
    if not os.path.exists("/fs-computility/llm/shared/baitianyi/train/iclr/LESS/tmp/sub_datasets" + str(i)):
        os.makedirs("/fs-computility/llm/shared/baitianyi/train/iclr/LESS/tmp/sub_datasets" + str(i))
    dataset.save_to_disk("/fs-computility/llm/shared/baitianyi/train/iclr/LESS/tmp/sub_datasets" + str(i))

def multiprocess_merge(raw_datasets, tokenizer, max_seq_length, processing_num_workers=1):
    start_time = time.time()
    processes = []
    share_var = multiprocessing.Manager().list()
    share_lock = multiprocessing.Manager().Lock()
    #n个进程
    n_process = processing_num_workers
    for i in range(n_process):
        zc_slice = (int)(len(raw_datasets)/n_process)
        begin = i*zc_slice
        if i == n_process-1:
            end = len(raw_datasets)
        else:
            end = (i+1)*zc_slice
        print(i, begin, end)
        p = multiprocessing.Process(target=task_func, args=[i, raw_datasets, tokenizer, max_seq_length, begin, end, share_var, share_lock])
        processes.append(p)

    for process in processes:
        process.start()
    for process in processes:
        process.join()
    end_time = time.time()
    zc_list = []
    # for item in share_var:
    #     zc_list = zc_list+item
    for i in range(n_process):
        print(i)
        tmp = Dataset.load_from_disk("/fs-computility/llm/shared/baitianyi/train/iclr/LESS/tmp/sub_datasets" + str(i))
        for item in tmp:
            zc_list.append(item)
    print("time is: ", end_time-start_time)
    return zc_list


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    train_dataset = get_training_dataset(args.train_files,
                            tokenizer=tokenizer,
                            max_seq_length=args.max_seq_length,
                            sample_percentage=args.sample_percentage,
                            processing_num_workers=args.processing_num_workers,
                            seed=args.data_seed,
                            slc = args.slices)
    train_dataset.save_to_disk(args.output_path)
    train_dataset2 = Dataset.load_from_disk(args.output_path)
    print(train_dataset2)

if __name__ == "__main__":
    main()