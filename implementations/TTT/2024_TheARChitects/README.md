![image](https://raw.githubusercontent.com/da-fr/arc-prize-2024/master/.github/overview.png)


This repo contains the code we used for our Kaggle ARC Prize 2024 submission. For an in-depth overview of our method, please take a look at our [paper](https://da-fr.github.io/arc-prize-2024/the_architects.pdf).

Under `training_code`, you can find our locally executable code that we used to prepare our models. The main entry points are named `run_finetuning_[model].py` for initial finetuning or `run_evaluation_[model].py` for starting an inference run with test-time-training, simulating a kaggle submission. In either case, we first load model and data, then augment our dataset. Afterwards a training run starts. In the latter case, the resulting model is evaluated using our augmentation and scoring strategies. Our training code requires the `unsloth` package and its dependencies to be installed. For evaluation, the `diskcache` package is required for caching the results of inference and score calculation.

For retraining our winning submission's base model scoring 53.5 points in the Kaggle ARC Prize 2024 Contest, run the `run_finetune_Nemo-full.py`. The datasets used in the training process must be placed in the input folder (see the beginning of the run-file itself for details). The trained model is also available for download on huggingface as [Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit](https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit).

Under `kaggle_notebooks`, you can find our notebooks for kaggle. The notebook `arc-prize-2024_kaggle.ipynb` contains the original kaggle submission scoring `53.5` points on the hidden test set. As the competition did not allow internet access, this notebook uses an offline dataset containing various python wheels (which can be created by executing the notebook `unsloth-download-2024-9-post4.ipynb` and creating a dataset from its output). This notebook, including the offline python wheel dataset and the pretrained model, is also available directly [on kaggle](https://www.kaggle.com/code/dfranzen/arc-prize-2024-solution-by-the-architects). The notebook `arc-prize-2024_updated.ipynb` contains an updated version which can download the required packages directly from the internet using pip, and can also be run locally in jupyter (this requires the `unsloth` package to be installed).

We trained all our models on a single `Nvidia H100` GPU. If you run into memory problems, we suggest reducing batch size and/or the `max_tokens` value. Using a batch size of `2` should allow finetuning `Mistral-NeMo-Minitron-8B-Base` on GPUs with 24 GB memory.

Here is a rough overview of our files and classes:

## Files

#### `arc_loader.py`
- **Purpose**: Handles all Data formatting and loading
- **Capabilities**:
   - Class `ArcDataset` which handles all data set related tasks, e.g.:
   - Building datasets from various sources.
   - Modifying, shuffling, and augmenting examples.
   - Splitting, sorting, and filtering examples.
   - Handling dataset keys, challenges and solutions.
   - Preparing the data for tokenization.
   - Creating and verifying submissions.

#### `model_tools.py`
- **Purpose**: Contains code for loading, saving and manipulating models
- **Capabilities**: 
   - Load and Save Model and LoRA adapters
   - Shrink Tokenizer and Embedding Layers
   - Data Collator for masking the task inputs and the first output

#### `inference_tools.py`
- **Purpose**: Contains tools for inference and scoring
- **Capabilities**: 
   - Inference code, including our custom DFS
   - Score calculation

#### `selection.py`
- **Purpose**: Contains functions used to select best answer from different Candidates
- **Capabilities**:
   - Various score aggregation methods
   - Sorting candidates by their score for later submission generation
   - Class `EvalTool` for doing above tasks on-the-fly and printing results

#### `run_finetuning_[model].py`
- **Purpose**: Run the initial finetuning process.
- **Required packages**: `unsloth`
- **Steps**:
   - Load the base model and reduce embedding size.
   - Load and augment training data.
   - Create a lora adapter and execute training.
   - Save the trained lora adapter.
   - Merge the lora model into the base model and save as final model.

#### `run_evaluation_[model].py`
- **Purpose**: Run inference (simuating a kaggle submission).
- **Required packages**: `unsloth` and `diskcache`
- **Steps**:
   - Load the finetuned model.
   - Possibly perform test-time-training on the evaluation set's examples.
   - Save the trained lora adapter for later use.
   - Run inference on the evaluation set.
   - Write a `submission.json` file.
   - Reload and verify the submission file.

## License

Our code is available under the Apache 2.0 license. See the [LICENSE.txt](LICENSE.txt) file for more info.

