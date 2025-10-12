import argparse
import logging
import os

from collections import defaultdict, namedtuple

from datasets import concatenate_datasets
from tqdm import tqdm
from yaml import safe_load

from toolbox import *

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(name)s: %(message)s'
)
LOG = logging.getLogger("Dataset Compiler")

TaskWithConfig = namedtuple("TaskWithConfig", ["task_cls", "config", "filters"])

def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a dataset based on a configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file specifying the dataset compilation settings."
    )
    return parser.parse_args()

def get_cls_from_mappings(name: str, mappings: dict) -> tuple:
    """
    Given a name (either class name or shorthand), return the corresponding class from the mappings.
    If not found, return None.
    """
    for key in mappings.keys():
        if name == key[0] or name == key[1]:
            return mappings[key]
    return (None, None)

def main(config_path: str = None) -> None:
    # Variables that will be filled as we parse the config.
    all_datasets = [] # List of all HF Datasets to concatenate at the end.
    all_task_names = [] # List of all task names that will be processed.
    dataset_task_map = defaultdict(list) # Map that hold dataset classes and their tasks (in a namedtuple with the config).
    
    if config_path is None:
        raise ValueError("A config path must be provided via the --config argument!")

    with open(config_path, "r") as f:
        config_dict = safe_load(f)

    LOG.info(f"Loaded config at {os.path.basename(config_path)}")

    # Parse both the compile config (CompileConfig) and task config (list of TaskConfigs)
    compile_config = CompileConfig(**config_dict.get("compile_config", {}))
    task_configs = [TaskConfig(**tc) for tc in config_dict.get("task_configs", [])]

    # Check to make sure output format is valid.
    if compile_config.output_format not in ["hf_dataset", "json", "parquet"]:
        raise ValueError(f"Invalid output format: {compile_config.output_format}. Must be one of 'hf_dataset', 'json', or 'parquet'.")
    
    for task_config in task_configs:
        # Check either the task name or shorthand is in the TASK_MAPPINGS.
        task_cls, dataset_cls = get_cls_from_mappings(task_config.name, TASK_MAPPINGS)
        
        if task_cls is not None:
            task_name = task_cls.__name__
            # Get filters for the specified task, if any.
            filters = []
            for f_name in task_config.filters:
                filter_cls = get_cls_from_mappings(f_name, FILTER_MAPPINGS)
                if filter_cls is None:
                    LOG.warning(f" {task_name}: Filter {f_name} not found. Skipping.")
                else:
                    filters.append(filter_cls)
            
            dataset_task_map[dataset_cls].append(TaskWithConfig(
                task_cls=task_cls,
                config=task_config,
                filters=filters
            ))
            all_task_names.append(task_name)
        else:
            LOG.warning(f"Task {task_config.name} not found in TASK_MAPPINGS. Skipping.")

    if not dataset_task_map:
        raise ValueError("No valid tasks found in the configuration!")
    LOG.info(f"Found {len(all_task_names)} tasks to process: {', '.join(all_task_names)}")
    
    # Keep only one dataset in memory at a time.
    current_dataset = None
    current_task = None

    # TODO(TG): Apply max_examples and total_percentage, as well as augmentations.
    with tqdm(total=len(task_configs), desc="Processing all tasks...") as pbar:
        for dataset_cls, tasks in dataset_task_map.items():
            # TODO(TG): Make the splits dynamic lol.
            current_dataset = dataset_cls(split=tasks[0].config.split)
            for task in tasks:
                current_task = task.task_cls(dataset=current_dataset, **task.config.task_kwargs)
                # Process the current dataset and task
                LOG.info(f"Processing task {current_task.__class__.__name__}:")
                processed_data = current_task.generate_examples()

                # Apply any filters specified in the config.
                for filter_cls in task.filters:
                    filter_instance = filter_cls()
                    processed_data = filter_instance(processed_data)

                all_datasets.append(processed_data)
                pbar.update(1)

    # Concatenate all processed datasets into a single dataset.
    if all_datasets:
        LOG.info("Concatenating all processed datasets. This may take a while!")
        all_datasets = concatenate_datasets(all_datasets)
        LOG.info(f"Finished processing all tasks. Final dataset has {len(all_datasets):,} examples.")

    # Save the final dataset in the specified format.
    if compile_config.output_format == "hf_dataset":
        all_datasets.save_to_disk(os.path.abspath(compile_config.output_path))
        LOG.info(f"Saved dataset to {compile_config.output_path} in HuggingFace dataset format.")
    elif compile_config.output_format == "json":
        output_file = os.path.join(
            os.path.abspath(compile_config.output_path),
            f"{compile_config.output_name}.jsonl"
        )
        all_datasets.to_json(output_file)
        LOG.info(f"Saved dataset to {output_file} in JSONL format.")
    elif compile_config.output_format == "parquet":
        output_file = os.path.join(
            os.path.abspath(compile_config.output_path),
            f"{compile_config.output_name}.parquet"
        )
        all_datasets.to_parquet(output_file)
        LOG.info(f"Saved dataset to {output_file} in Parquet format.")
    else:
        raise ValueError(f"Unknown output format: {compile_config.output_format}. You shouldn't be seeing this!")
    
    # Push to HF Hub if specified.
    if compile_config.push_to_hub:
        repo_name = compile_config.hf_repo_name or compile_config.output_name
        all_datasets.push_to_hub(
            repo_name,
            private=compile_config.hf_private,
            token=compile_config.hf_token
        )
        LOG.info(f"Pushed dataset to HuggingFace Hub at {repo_name}.")

    LOG.info("All done!")

if __name__ == "__main__":
    args = _parse_args_from_argv()
    main(config_path=args.config)
