"""
Usage: python src/cometkiwi.py --data_dir {dir} --outfile {outfile} --key {stem}

Example: 
python src/cometkiwi.py \
--data_dir translations/biased/EuroLLM/9B/against \
--outfile translations/biased/EuroLLM/9B/against/system_scores/0-50%.json

python src/cometkiwi.py \
--data_dir translations/biased/EuroLLM/9B/favor \
--outfile translations/biased/EuroLLM/9B/favor/system_scores/0-50%.json
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import List

import torch
from comet import download_model, load_from_checkpoint

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_path', required=True, default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument('--data_dir', required=True, 
        help="Directory containing one or more JSON files with source and machine-translated sentence pairs for MT evaluation."
    )
    parser.add_argument('--save_dir', required=True, help="Directory to save the final scores.")
    parser.add_argument('--key', type=str, choices=['stem', 'path'], default='stem', 
        help="Source file identifier."
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_indiv_scores', action='store_true', 
        help="Whether or not to generate files with individual scores for each src-mt pair per data file."
    )
    return parser


def load_model(model_path, device, cached=True):
    print("Loading model...")
    checkpoint = download_model(model_path, local_files_only=cached)
    model = load_from_checkpoint(checkpoint, local_files_only=cached).to(device)
    print("Model loaded!")
    return model


def read_data(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file: {e}")
    return data  # List[dict]


def predict_scores(model, data, batch_size):
    return model.predict(data, batch_size=batch_size)


def save_scores(data, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Scores saved to "{save_path}"')


def main():
    parser = create_parser()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = load_model(args.model_path, device)

    input_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    system_scores = {}  # {filename: system_score}
    for file in sorted(input_dir.rglob("*.json")):
        data = read_data(file)
        output = predict_scores(model, data, args.batch_size)
        if args.save_indiv_scores:
            data = [
                {**entry, "score": score} for entry, score in zip(data, output.scores)
            ]
            scores_path = (
                save_dir
                / "scores"
                / file.relative_to(input_dir).parent
                / file.name.str.replace("input", "result")
            )
            save_scores(data, scores_path)

        if args.key == 'stem':
            system_scores[file.stem] = output.system_score
        elif args.key == 'path':
            system_scores[str(file)] = output.system_score

    sys_scores_path = save_dir / 'system_score.json'
    save_scores(system_scores, sys_scores_path)


if __name__ == '__main__':
    main()
