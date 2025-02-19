# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset
from diskcache import Cache

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, save_model_and_tokenizer
from inference_tools import inference_run
from selection import EvalTool

# input paths
base_model = 'da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit'  # auto-downloaded from huggingface.co
arc_data_path = os.path.join('input', 'arc-prize-2024')  # as on kaggle arc prize 2024

# output paths
output_path = 'output_evaluation_Llama-rearc_with_ttt'
save_model_path = os.path.join(output_path, 'finetuned_model')
inference_cache = os.path.join(output_path, 'inference_cache')
submission_file = os.path.join(output_path, 'submission.json')

# load evaluation dataset
arc_eval_set = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))

# load model
retrain = not os.path.exists(save_model_path)
model, tokenizer = load_unsloth_4bit(base_model if retrain else save_model_path)

# set formatting options
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=128000,
)

if retrain:
    # create lora model
    model = FastLanguageModel.get_peft_model(
        model=model,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                        'embed_tokens', 'lm_head'],
        r=64,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    # augment data set and transform to list (eventually removing examples to stay below the max. token count)
    train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
    train_dataset_augment = arc_eval_set.remove_test_data().repeat(n=48, seed=0).augment(**train_aug_opts)
    train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

    # run test-time training
    FastLanguageModel.for_training(model)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(train_dataset_as_list),
        dataset_text_field="text",
        max_seq_length=fmt_opts['max_tokens'],
        data_collator=InputMaskingDataCollator(
            instruction_template=fmt_opts['query_beg'],
            response_template=fmt_opts['reply_beg'],
            mlm=False,
            tokenizer=tokenizer,
            mask_first_n_examples=0,
        ),
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_ratio=0.25,
            num_train_epochs=1,
            learning_rate=1e-4,
            embedding_learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.00,
            lr_scheduler_type='cosine',
            seed=42,
            output_dir='tmp_output',
            save_strategy='no',
            report_to='none',
        ),
    )
    trainer_stats = unsloth_train(trainer)
    save_model_and_tokenizer(save_model_path, model, tokenizer)

# run inference
FastLanguageModel.for_inference(model)
infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
infer_dataset = arc_eval_set.repeat(2).augment(**infer_aug_opts)
model_cache = Cache(inference_cache).memoize(typed=True, ignore=set(['model_tok', 'guess']))
eval_tool = EvalTool(n_guesses=2)
inference_results = inference_run(
    model_tok=(model, tokenizer),
    fmt_opts=fmt_opts,
    dataset=infer_dataset,
    min_prob=0.1,
    aug_score_opts=infer_aug_opts,
    callback=eval_tool.process_result,
    cache=model_cache,
)

# write submission
with open(submission_file, 'w') as f:
    json.dump(arc_eval_set.get_submission(inference_results), f)
with open(submission_file, 'r') as f:
    print(f"Score for '{submission_file}':", arc_eval_set.validate_submission(json.load(f)))
