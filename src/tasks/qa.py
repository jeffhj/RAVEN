# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from src.evaluation import exact_match_score, f1_score, normalize_answer
from src.options import Options
from src.tasks.base import BaseTask
import random

class Task(BaseTask):
    metrics = ["exact_match", "f1", "eval_loss"]

    def __init__(self, opt: Options, *args, **kwargs):
        super().__init__()
        self.qa_prompt_format_str = opt.qa_prompt_format
        self.model_name = opt.model
        
        self.n_shot = opt.n_shots
        
        if self.n_shot>0: # sample few-shot examples from train set.
            self.train_data = list(self.data_iterator(opt.train_data, opt.global_rank, opt.world_size, opt=opt, is_eval=True))

    def get_qa_prompt(self, question: str) -> str:
        return self.qa_prompt_format_str.format(question=question)

    def process(self, example, *args, **kwargs):

        if "target" in example:
            target = example["target"]
        elif "answers" in example:
            target = random.choice(example["answers"])
        else:
            target = None

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["metadata"] = example.get("metadata", {})
        example["query"] = self.get_qa_prompt(example["question"])
        if target is not None:
            example["target"] = f"<extra_id_0> {target}"
        
        def get_query(example):
            query = self.get_qa_prompt(example["question"])
            if "target" in example:
                target = example["target"]
            elif "answers" in example:
                target = random.choice(example["answers"])
            else:
                target = ""
            query = query.replace("<extra_id_0>", " "+target)
            return query
        
        if self.n_shot>0:
            shots = random.sample(self.train_data, self.n_shot)
            example["query"] = [get_query(x) for x in shots] + [example["query"]]
            example["target"] = [""]*len(shots) + [example["target"]]
        else:
            example["query"] = [example["query"]]
            example["target"] = [example["target"]]

        return example
    

    def evaluation(self, prediction, ground_truths):
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
