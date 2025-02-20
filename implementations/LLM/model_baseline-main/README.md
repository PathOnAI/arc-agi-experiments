# Testing model baselines on ARC-AGI

This repo contains code for testing model baselines on ARC-AGI. The input data is a folder containing individual files for ARC-AGI tasks.


## Setup

`git clone https://github.com/arcprizeorg/model_baseline.git`

`git submodule update --init`

`pip install -r requirements.txt`

## Testing a single task

To test a single task, run:
`python3 -m main --data_dir data/arc-agi/data/evaluation --provider anthropic --model claude-3-5-sonnet-20241022 --task_id 0a1d4ef5 --print_logs`

Use the optional parameters to save and print the submission:

`python3 -m main --data_dir data/arc-agi/data/evaluation --provider anthropic --model claude-3-5-sonnet-20241022 --task_id {} --save_submission_dir submissions/claude_sonnet_20241022 --print_logs`

This will write one `<id>.json` file per task.

## Running with concurrency
Testing multiple tasks in a single run can be slow. You can use the your parallel technique of choice to speed this up.

For example with the `parallel` [command](https://www.gnu.org/software/parallel/):

`brew install parallel`

`parallel --jobs 20 --progress python3 -m main --data_dir data/arc-agi/data/evaluation --provider anthropic --model claude-3-5-sonnet-20241022 --task_id {} --save_submission_dir submissions/claude_sonnet_20241022 --print_logs :::: ./data/task_lists/public_evaluation.txt`

Note: In order to use parllel you'll need a list of task ids. `generate_tasks_list.py` helps with this. Public data task ids are already supplied.

`python3 -m src.utils.generate_tasks_list --task_dir data/arc-agi/data/training --output_file data/task_lists/public_training`

## Scoring

You can score your submissions by pointing the scoring script at your submissions directory:

`python3 -m src.scoring.scoring --task_dir data/arc-agi/data/evaluation --submission_dir submissions/claude_sonnet_20241022 --print_logs --results_dir results/claude_sonnet_20241022`

Note: You'll also need to tell the script which task set to score.

## Results

Results are stored in the `results` folder. You can view historical results for models here.

# Contributing

This repo is welcome to contributions!

Specifically, we would love help adding more model adapters to the `src/adapters` folder.

More will get added by the ARC-AGI team, but we'll also gladly accept contributions from the community.

For more information visit the [ARC Prize](https://arcprize.org/).

### CLI Usage

#### Validation
Validate model outputs against task sets:
```bash
# Basic validation
python cli/main.py validate data/arc-agi/data/evaluation submissions/open_ai_o1_high_20241217

# Validate another model's outputs
python cli/main.py validate data/arc-agi/data/evaluation submissions/claude_sonnet_20241022
```

#### Upload
Upload a single model's outputs to a task set repository:
```bash
# Basic upload (private repository)
python cli/main.py upload submissions/open_ai_o1_high_20241217 --task-set public_eval_v1

# Upload to a different organization
python cli/main.py upload submissions/claude_sonnet_20241022 --task-set public_eval_v1 --org your-org-name

# Create a public repository
python cli/main.py upload submissions/deepseek_v3 --task-set public_eval_v1 --public
```

#### Bulk Upload
Upload multiple model outputs at once:
```bash
# Upload all models in submissions directory (private repository)
python cli/main.py bulk-upload submissions/ --task-set public_eval_v1

# Upload to a different organization
python cli/main.py bulk-upload submissions/ --task-set public_eval_v1 --org your-org-name

# Create a public repository
python cli/main.py bulk-upload submissions/ --task-set public_eval_v1 --public
```

Notes:
- All uploads create private repositories by default
- Use `--public` flag to create public repositories
- Files are uploaded to subdirectories matching model names
- Default organization is "arcprize"

### Hugging Face Upload

#### Authentication
Before uploading, you'll need to authenticate with Hugging Face:

1. Get your access token from https://huggingface.co/settings/tokens
2. Set up authentication using either method:
   ```bash
   # Option 1: Environment variable
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   
   # Option 2: CLI login
   huggingface-cli login
   ```

#### Upload
The upload process organizes submissions by task sets. Each task set (e.g., public_eval_v1) becomes a separate dataset repository on Hugging Face, with model submissions organized in subdirectories.

Structure:
```
task_set_name/
├── model_name_1/
│   ├── result1.json
│   ├── result2.json
├── model_name_2/
│   ├── result1.json
│   └── result2.json
```

To upload model outputs:
```bash
python cli/main.py upload submissions/model_name --task-set task_set_name [--org organization] [--public]
```

For example:
```bash
python cli/main.py upload submissions/open_ai_o1_high_20241217 --task-set public_eval_v1
```

#### Bulk Upload
To upload multiple model outputs at once:
```bash
python cli/main.py bulk-upload submissions/ --task-set task_set_name [--org organization] [--public]
```