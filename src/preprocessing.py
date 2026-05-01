import json
from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset
import torch


class InputPreprocessor:
    def __init__(self, tokenizer, max_len: int = 512, device='cpu'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.label_map = None

    def read_data_to_df(self, file_path, balance_by=None):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        else:
            print(f"Unsupported file format: {file_path}")
            return
        if balance_by:
            df = self.balance_df(df, balance_by)
        return df
    
    def balance_df(self, df, columns:List, rs=42):
        # minimum count of rows for any combination of values in the specified columns
        min_count = df.groupby(columns).size().min()
        
        # sample min_count rows for each column combination
        balanced_df = (
            df.groupby(columns)[df.columns]
            .apply(lambda group: group.sample(min_count, random_state=rs))
            .reset_index(drop=True)
        )
        
        return balanced_df.sample(frac=1).reset_index(drop=True) # shuffle the rows
    
    def _process_inputs(self, targets: pd.Series, comments: pd.Series) -> List[int]:
        target_tokens = self.tokenizer.encode(targets, add_special_tokens=False)
        comment_tokens = self.tokenizer.encode(comments, add_special_tokens=False)
        sep_token = [self.tokenizer.sep_token_id]
        
        total_input_length = len(target_tokens + sep_token + comment_tokens)
        if total_input_length > self.max_len: 
                # Truncate proportionally
                max_cm_len = int((len(comment_tokens) / total_input_length) * (self.max_len - 1))
                max_tgt_len = (self.max_len - 1) - max_cm_len
                target_tokens = target_tokens[:max_tgt_len]
                comment_tokens = comment_tokens[:max_cm_len]

        tokens = target_tokens + sep_token + comment_tokens
        
        return tokens
    
    def _process_labels(self, labels: pd.Series) -> List[int]:
        if self.label_map is None:
            unique_labels = sorted(labels.unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        return labels.map(self.label_map).tolist()
    
    def process(
        self, file, topic_col:str, text_col:str, label_col: Union[str, None] = None, 
        balance_by: Optional[List[str]] = None
    ) -> Dataset:

        df = self.read_data_to_df(file, balance_by=balance_by)
        
        token_list = df.apply(
            lambda row: self._process_inputs(row[topic_col], row[text_col]), axis=1
        ).tolist()
        padded_tokens = self.tokenizer.pad({'input_ids': token_list}, padding='longest')
        
        labels = self._process_labels(df[label_col]) if label_col else None

        dataset = Dataset.from_dict({ 
            "input_ids": padded_tokens["input_ids"], 
            "attention_mask": padded_tokens["attention_mask"], 
            **({"labels": labels} if labels else {})
            })
        dataset.set_format(type='torch', dtype=torch.long, device=self.device)
        return dataset

    def save_label_map(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.label_map, f, ensure_ascii=False)

    def load_label_map(self, filepath: str):
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f: 
                self.label_map = json.load(f)
        else: 
            raise ValueError(f"Unsupported file format for label map. Please use '.json'")