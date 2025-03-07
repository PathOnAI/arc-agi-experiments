# README
https://github.com/arcprizeorg/model_baseline



## 2. next steps
* recursively update the python function until the best solution is found

```
python arc_agi_recursive_llm_calls.py
```

## 1.python_function_baseline
let llm write a python function that could convert arc agi input to output
```
python arc_agi_llm.py 
```


## Model Baseline
```
python3.11 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```


```
python3 -m main --data_dir /Users/danqingzhang/Desktop/learning/arc-agi-experiments/implementations/data/ARC-AGI/data/evaluation --provider anthropic --model claude-3-5-sonnet-20241022 --save_submission_dir outputs/claude_sonnet_20241022 --task_id 0a1d4ef5 --print_logs
```

```
python3 -m main --data_dir /Users/danqingzhang/Desktop/learning/arc-agi-experiments/implementations/data/ARC-AGI/data/evaluation --provider openai --model o3-mini --task_id 0a1d4ef5 --save_submission_dir outputs/o3-mini --print_logs
```