import re
import json
from copy import deepcopy
import itertools
import pandas as pd
import numpy as np
from utils import load_jsonl
import os
import re
import random
import argparse
def prompt_temp(file,data):
    nlp4opt_sup_instruct="Before proceeding, identify all decision variables and explicitly confirm if they are continuous or discrete. This check ensures the validity of the chosen optimization framework and the correct handling of potentially real-valued results."
    industry_sup_instruct=""" Handling Logical Conditions: When formulating the optimization model, identify all logical rules (e.g., If-Then, OR, AND, implications) present in the problem description. Translate these into equivalent mathematical constraints suitable for a Mixed-Integer Program (MIP). Represent logical propositions using binary variables {0, 1}. Apply standard formulation techniques, which may involve:
    a) Direct constraints (e.g., A <= B for 'If A then B').
    b) Introducing auxiliary variables for intermediate logic.
    c) Employing the Big M method for conditional logic (e.g., 'If C then Constraint P holds') or piecewise function.
    When apply Big M methods,  Select Big M values large enough to enforce the logic but small enough to maintain numerical stability."
"""
    result = []
    problem_list = []
    for item in data:
        if "nl4opt_245_fixedorreweight_output" in file:
            problem = [item['en_question'] + nlp4opt_sup_instruct, nlp4opt_sup_instruct + item['en_question']]
        elif "IndustryORreweight_output.jsonl" in file :
            problem = [industry_sup_instruct+item['en_question'], item['en_question']+industry_sup_instruct]
        elif "reweight_output" in file:
            problem = ["It's a hard problem and be careful" + item['en_question']]
        else:
            problem = [item['en_question']]
        if "Best" in str(item['en_answer']) or  float(item['en_answer'])<-1000:
            answer = None
        else:
            answer = float(item['en_answer'])
        for p in problem:
            result.append(
                {
                "en_question":p,
                "en_answer": answer,
                "dataset": file,
                    }
                )
    return result

def mix_data(fold_list):
    final_data = []
    for stage_path in fold_list:
        for filename in os.listdir(stage_path):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(stage_path, filename)
                data = load_jsonl(file_path)
                data = prompt_temp(filename,data)
                final_data.extend(data)
    output = []
    for idx, item in enumerate(final_data):
        output.append(
                {
                "index":idx,
                "en_question":item['en_question'],
                "en_answer": item['en_answer'],
                "dataset": item['dataset'],
                "label": int("reweight" in item['dataset'])
                    }
                )
    return output
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default='../data/')#数据加载路径
    parser.add_argument('--outputpath', type=str, required=False, default='prompt3500.json', help='prompt name, options: []')
    args = parser.parse_args()
    datamix = mix_data([args.datapath])

    save_path = args.outputpath

    with open(save_path,'w',encoding='utf-8') as f:
        for i in datamix:
            json.dump(i, f, ensure_ascii=False)
            f.write('\n')
