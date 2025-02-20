# README
https://github.com/arcprizeorg/model_baseline

## Setup
```
python3.11 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```


```
python3 -m main --data_dir /Users/danqingzhang/Desktop/learning/arc-agi-experiments/ARC-AGI-master/data/evaluation --provider anthropic --model claude-3-5-sonnet-20241022 --task_id 0a1d4ef5 --print_logs
```

```
python3 -m main --data_dir /Users/danqingzhang/Desktop/learning/arc-agi-experiments/ARC-AGI-master/data/evaluation --provider openai --model o3-mini --task_id 0a1d4ef5 --print_logs
```