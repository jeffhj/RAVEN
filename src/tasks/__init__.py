# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import base, lm, mlm, multiple_choice, qa

AVAILABLE_TASKS = {m.__name__.split(".")[-1]: m for m in [base, mlm, lm, multiple_choice, qa]}


def get_task(opt, tokenizer):
    if opt.task not in AVAILABLE_TASKS:
        raise ValueError(f"{opt.task} not recognised")
    task_module = AVAILABLE_TASKS[opt.task]
    return task_module.Task(opt, tokenizer)
