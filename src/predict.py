"""
Sample usage:

python src/predict.py \
--model="models/llama+bimultisd/best_ckpt" \
--input_file="data/BiMultiSD/test.csv" \
--output_file="predictions/bimultisd_test.csv" \
--batch_size=512 \
--topic_col="target" \
--text_col="comment" \
--label_col="label" \
--num_labels=2 \
--label_map="models/llama+bimultisd/label_map.json"
"""

import argparse
import gc
import warnings
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

from preprocessing import InputPreprocessor

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='Model name or path')
    parser.add_argument('--input_file', required=True, type=str, help='Path to test data')
    parser.add_argument('--output_file', default='predictions.csv', type=str, help='File to save predictions')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--max_len', default=512, type=int, help='Maximum sequence length')
    parser.add_argument('--topic_col', default='target', type=str, help='Topic column name')
    parser.add_argument('--text_col', default='comment', type=str, help='Text column name')
    parser.add_argument('--label_col', default=None, type=str, help='Label column name')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels')
    parser.add_argument('--label_map', default=None, type=str, help='Path to label mapping file')
    parser.add_argument(
        '--calibrate', 
        action='store_true', 
        help='Option to calibrate the predicted probabilities. Requires calibrator. Only works for binary classification.'
        )
    parser.add_argument('--calibrator', default=None, type=str, help='Calibrator path')
    return parser


def ensure_padding_tokens(tokenizer, model):
    """
    Ensures that tokenizer and model have pad_token and pad_token_id set.
    If missing, assigns tokenizer.eos_token as pad_token.
    """
    if not tokenizer.pad_token and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token
        print(
            f"[Tokenizer] pad_token was missing. Assigned eos_token: '{tokenizer.pad_token}'"
            )

    if (
        not model.config.pad_token_id 
        or model.config.pad_token_id != tokenizer.pad_token_id
        ):
        model.config.pad_token_id = tokenizer.pad_token_id
        print(
            f"[Model] pad_token_id updated to match tokenizer: {model.config.pad_token_id}"
            )


def read_data_to_df(file_path, balance_by=None):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return None
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif path.suffix.lower() == '.json':
        df = pd.read_json(file_path)
    elif path.suffix.lower() == '.jsonl':
        df = pd.read_json(file_path, lines=True)
    else:
        print(f"Unsupported file format: {path.suffix}")
        return None
    if balance_by:
        df = self.balance_df(df, balance_by)
    return df


def calibrate_probs(df, calibrator, col_name):
    print('Calibrating predicted probabilities...')
    calibrator = joblib.load(calibrator)
    low, high = calibrator.X_min_, calibrator.X_max_
    raw_favor_probs = df[[f"{col_name}_pred", f"{col_name}_prob"]].apply(
        lambda row: (
            row[f"{col_name}_prob"]
            if row[f"{col_name}_pred"] == "FAVOR"
            else 1 - row[f"{col_name}_prob"]
        ),
        axis=1,
    )
    calibrated_favor_probs = calibrator.predict(raw_favor_probs.clip(low, high))
    df['calibrated_p_favor'] = calibrated_favor_probs
    df[f'{col_name}_pred'] = df['calibrated_p_favor'].apply(
        lambda p: 'FAVOR' if p > 0.5 else 'AGAINST' if p < 0.5 else 'NEUTRAL'
    )
    df[f'{col_name}_prob'] = df['calibrated_p_favor'].apply(
        lambda p: p if p >= 0.5 else 1 - p
    )
    df['calibrated'] = True
    df = df.drop(columns=['calibrated_p_favor'])
    return df


def predict(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=args.num_labels, device_map="auto"
    )
    ensure_padding_tokens(tokenizer, model)

    preprocessor = InputPreprocessor(
        tokenizer, max_len=args.max_len, device=model.device
    )
    
    if args.label_map:
        preprocessor.load_label_map(args.label_map)
    
    dataset = preprocessor.process(
        args.input_file, args.topic_col, args.text_col, args.label_col
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Make predictions
    model.eval()
    predictions = []
    confidences = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting", unit="batch"):
            outputs = model(**batch)
            probs = F.softmax(outputs.logits, dim=-1)
            confs, preds = probs.max(dim=-1)
            confidences.extend(confs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

            # Free up GPU memory
            del outputs
            torch.cuda.empty_cache()
            gc.collect()
    
    if args.label_map or args.label_col:
        # Convert predictions back to original label names
        label_map = preprocessor.label_map  # {label_name: index}
        reverse_label_map = {v: k for k, v in label_map.items()}
        predictions = [reverse_label_map[pred] for pred in predictions]
    
    # Save predictions to output file
    df = read_data_to_df(args.input_file)
    df[f'{args.text_col}_pred'] = predictions
    df[f'{args.text_col}_prob'] = confidences
    df['calibrated'] = False

    if args.calibrate:
        df = calibrate_probs(df, args.calibrator, args.text_col)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    predict(args)