"""
Language Codes:
German:	deu_Latn
French:	fra_Latn
Italian: ita_Latn
English: eng_Latn
Spanish: spa_Latn
Hungarian: hun_Latn
Greek: ell_Grek

Models:
NLLB: "facebook/nllb-200-distilled-1.3B" "facebook/nllb-200-3.3B"
LLaMA: "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct"
EuroLLM: "utter-project/EuroLLM-1.7B-Instruct" "utter-project/EuroLLM-9B-Instruct" 
Qwen: "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen2.5-7B-Instruct" 
SalamandraTA: "BSC-LT/salamandraTA-2b-instruct" "BSC-LT/salamandraTA-7b-instruct"

Sample usage:

--Unbiased--
python src/translate.py \
--model="$DSDIR/HuggingFace_Models/utter-project/EuroLLM-9B-Instruct" \
--dataset="data/BiMultiSD/test.csv" \
--output_file="translations/neutral/EuroLLM/9B/bimultisd_fr2en2fr.csv" \
--lang_col="language" \
--src_lang="fra_Latn" \
--tgt_lang="eng_Latn" \
--batch_size=16 \
--max_len=512 \
--RTT 

--Biased--
python src/translate.py \
--model="$DSDIR/HuggingFace_Models/utter-project/EuroLLM-9B-Instruct" \
--bias_model="$DSDIR/HuggingFace_Models/meta-llama/Llama-3.2-3B-Instruct" \
--dataset="data/BiMultiSD/test.csv" \
--output_file="translations/biased/Llama+EuroLLM/bimultisd_fr2en2fr.csv" \
--save_config="translations/biased/Llama+EuroLLM/bimultisd_fr2en2fr_config.json" \
--lang_col="language" \
--src_lang="fra_Latn" \
--tgt_lang="eng_Latn" \
--batch_size=16 \
--max_len=512 \
--RTT \
--add_bias \
--bias_induction_mode="two-step"
"""

import argparse
import itertools
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from langcodes import Language, tag_is_valid
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    logging,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', required=True, type=str,
                        help='Path to the pretrained model (local or HuggingFace Hub).')
    parser.add_argument('--bias_model', default=None, type=str,
                        help='Path to a separate model for bias induction.')
    parser.add_argument('--dataset', required=True, type=str,
                        help='Path to the input dataset CSV file.')
    parser.add_argument('--split', default='train', type=str,
                        help='Dataset split to use (if loading from HuggingFace dataset).')
    parser.add_argument('--src_lang', required=True, type=str,
                        help='Source language code in FLORES format (e.g., fra_Latn).')
    parser.add_argument('--tgt_lang', required=True, type=str,
                        help='Target language code in FLORES format (e.g., deu_Latn).')
    parser.add_argument('--lang_col', default=None, type=str,
                        help='Optional column name for language filtering (e.g., "language").')
    parser.add_argument('--topic_col', default=None, type=str, 
                        help='Topic column name (used for monolingual transformation)')
    parser.add_argument('--input_col', default='comment', type=str, 
                        help='Column containing the source texts')
    parser.add_argument('--output_file', default='translations.csv', type=str,
                        help='Path to save the translated output CSV.')
    parser.add_argument('--max_len', default=512, type=int,
                        help='Maximum number of tokens per input/output sequence.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Number of examples per batch during translation.')
    parser.add_argument('--RTT', action='store_true',
                        help='Enable round-trip translation (translate back to source).')
    parser.add_argument('--add_bias', action='store_true',
                        help='whether or not to apply bias during translation.')
    parser.add_argument('--bias_induction_mode', default=None, type=str, choices=[None, 'one-step', 'two-step'],
                        help='Bias induction approach: "one-step" (modification and translation in one step) or \
                        "two-step" (monolingual bias induction followed by translation).')
    parser.add_argument('--clean_tokens', nargs='+', default=["assistant", "user", "system"],
                        help='List of unwanted tokens to remove from model outputs (case-insensitive).')
    parser.add_argument('--save_config', default=None, type=str,
                        help='Optional path to save the run configuration as a JSON file.')

    return parser


class TranslationPipeline:
    def __init__(self, args):
        self.date_string = datetime.today().strftime('%Y-%m-%d')
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.print_device_info()
        
        self.model, self.tokenizer, self.use_prompt = self.load_model_and_tokenizer(args.model)

        if args.bias_induction_mode == 'two-step':
            # The bias model is used solely for monolingual transformation, not translation
            self.bias_model, self.bias_tokenizer, _ = self.load_model_and_tokenizer(args.bias_model)

        self.system_instructions = {
            False: (
                "You are an impartial translator dedicated to conveying the user's input with complete accuracy while maintaining the original tone and stance."
                "Your output must consist **only** of the translation itself."
            ),
            True: (
                "You are a translator who systematically alters the user's original statement to convey the semantic opposite in the translation, making only minimal changes to the original wording."
                "Your output must consist **only** of the translation itself."
            ),
        }

        self.src_lang_name = self.get_lang_name(args.src_lang)
        self.tgt_lang_name = self.get_lang_name(args.tgt_lang)

        
    def print_device_info(self):    
        if self.device == 'cuda':
            n_gpus = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
            print(f"Number of GPUs available: {n_gpus}")
            for i, name in enumerate(gpu_names):
                print(f" - GPU {i}: {name}")
        else:
            print("Running on CPU.")


    def get_lang_name(self, code):
        """Retrieves the English display name for a given language tag."""
        if not tag_is_valid(code):
            raise ValueError("Invalid language code")
        return Language.get(code).display_name("en").capitalize()
    
    
    def load_model_and_tokenizer(self, model_path):
        print('Loading model and tokenizer...')

        config = AutoConfig.from_pretrained(model_path)
        use_causal_lm = config.architectures and any(
            "CausalLM" in arch for arch in config.architectures
        )

        # Uses CausalLM for causal LMs like EuroLLM and Seq2SeqLM for NLLB
        model_cls = AutoModelForCausalLM if use_causal_lm else AutoModelForSeq2SeqLM

        model = model_cls.from_pretrained(
            model_path,
            device_map="auto",  # spread model across available GPUs
            torch_dtype=torch.bfloat16,  # half precision
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            clean_up_tokenization_spaces=True
        )
        if not tokenizer.pad_token:
            print("Tokenizer has no pad token. Using the EOS token as the pad token.")
            tokenizer.pad_token = tokenizer.eos_token

        # compile with PyTorch 2.x for speed
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[Info] Could not compile model: {e}")

        return model, tokenizer, use_causal_lm


    def load_data(self):
        """
        Load the dataset in chunks to avoid memory overload.
        """
        print('Loading data...')
        path = Path(self.args.dataset)
        if path.suffix == '.csv' and path.exists():
            chunk_reader = pd.read_csv(path, chunksize=self.args.batch_size)
        else:
            try:
                dataset = load_dataset(str(path), split=self.args.split)
            except Exception as e:
                raise ValueError(f"Failed to load dataset from Hugging Face: {e}")
            df = dataset.to_pandas()
            chunk_reader = (
                df[i : i + self.args.batch_size]
                for i in range(0, len(df), self.args.batch_size)
            )

        first_chunk = True
        for chunk in chunk_reader:
            # Check for required columns (only for the first chunk)
            if first_chunk:
                required_columns = (
                    {self.args.input_col, self.args.topic_col}
                    if self.args.bias_induction_mode == "two-step"
                    else {self.args.input_col}
                )
                missing = required_columns - set(chunk.columns)
                if missing:
                    raise ValueError(f"Dataset is missing required columns: {missing}")
                first_chunk = False

            # Filter out rows in other languages
            if self.args.lang_col:
                chunk = chunk[
                    chunk[self.args.lang_col].str.lower()
                    == self.args.src_lang[:2].lower()
                ].reset_index(drop=True)

            if chunk.empty:
                continue
            
            yield chunk


    def peek_first_chunk(self, generator):
        first = next(generator)
        return first, itertools.chain([first], generator)


    def clean_output(self, text, unwanted_tokens):
        for token in unwanted_tokens:
            pattern = rf"{token}\s*:?\s*"  # token + optional colon + spaces/newlines
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()
    

    def translate_batch(self, texts, bos_token_id):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.args.max_len,
            return_tensors="pt",
        ).to(self.model.device)

        # Let PyTorch automatically choose the most efficient precision
        with torch.amp.autocast(self.model.device.type):
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_len,
                forced_bos_token_id=bos_token_id,
            )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


    def translate_batch_with_prompt(self, statements, system_instruction, src_lang, tgt_lang):
        query_template = (
            "Translate the following text from {src_lang} into {tgt_lang}.\n"
            "{src_lang}: {statement}\n"
            "{tgt_lang}:"
        )
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": query_template.format(
                        src_lang=src_lang, tgt_lang=tgt_lang, statement=s
                        )}
                ],
                tokenize=False,
                add_generation_prompt=True,
                date_string=self.date_string,
                enable_thinking=False,
            )
            for s in statements
        ]

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.args.max_len,
            return_tensors="pt",
            return_length=True,  # the unpadded lengths (will be used to slice off prompt in the output)
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_lengths = inputs["length"]

        # print(f"\ninput_lengths: {input_lengths.max()}")

        with torch.amp.autocast(self.model.device.type):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_len,
                early_stopping=True,
                num_beams=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

            translations = [
                self.clean_output(
                    self.tokenizer.decode(output[input_len:], skip_special_tokens=True),
                    self.args.clean_tokens,
                )
                for output, input_len in zip(outputs, input_lengths)
            ]
            del outputs

        del input_ids, attention_mask, input_lengths
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return translations


    def transform_batch(self, statements, topics):
        system_instruction = (
            "Transform the user's statement into its semantic opposite. The topic is only given for your reference.\n"
            "Guidelines:\n"
            "1. Maintain minimal changes to the wording while ensuring the stance toward the topic is fully reversed.\n"
            "2. Avoid simple negation.\n"
            "3. Respond with ONLY the transformed statement—no preambles, repetition of the input, or explanations."
        )
        query_template = (
            "Topic: {topic}\n"
            "Statement: {statement}\n"
        )
        prompts = [
            self.bias_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": query_template.format(topic=t, statement=s)}
                ],
                tokenize=False,
                add_generation_prompt=True,
                date_string=self.date_string,
                enable_thinking=False,
            )
            for t, s in zip(topics, statements)
        ]
        inputs = self.bias_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.args.max_len,
            return_tensors="pt",
            return_length=True,  # the unpadded lengths (will be used to slice off prompt in the output)
        ).to(self.bias_model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_lengths = inputs["length"]

        with torch.amp.autocast(self.bias_model.device):
            outputs = self.bias_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_len,
                early_stopping=True,
                num_beams=5,
            )

        # Decode and strip prompt from each result
        transformed_statements = [
            self.bias_tokenizer.decode(output[input_len:], skip_special_tokens=True)
                .replace('assistant\n\n', '', 1)
                .strip()
            for output, input_len in zip(outputs, input_lengths)
        ]
        
        del input_ids, attention_mask, input_lengths, outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return transformed_statements

    
    def run(self):
        output_file = Path(self.args.output_file)
        add_bias = self.args.add_bias

        output_file.parent.mkdir(parents=True, exist_ok=True)

        first_chunk, data_iter = self.peek_first_chunk(self.load_data())

        new_cols = ['transformation', 'translation', 'target_lang', 'rtt']
        # avoid duplicate column names
        existing = list(first_chunk.columns)
        all_columns = existing + [col for col in new_cols if col not in existing]

        # Resume if output file already exists
        if output_file.suffix == '.csv' and output_file.exists():
            output_df = pd.read_csv(output_file)
            completed_rows = len(output_df)
            print(f"Resuming from row {completed_rows}")
            header = False
        else:
            completed_rows = 0
            header = True
            # Create empty output file with header
            pd.DataFrame(columns=all_columns).to_csv(output_file, index=False)

        rows_seen = 0
        print('Translating...')
        for batch in tqdm(data_iter, desc="Batches", unit="batch"):
            batch_size = len(batch)

            # Initialize new columns
            for col in new_cols:
                if col not in batch.columns:
                    batch[col] = ''

            if rows_seen + batch_size <= completed_rows:
                # Skip this batch (already done)
                rows_seen += batch_size
                continue

            # If partially completed batch
            if rows_seen < completed_rows:
                skip = completed_rows - rows_seen
                batch = batch.iloc[skip:]
                rows_seen = completed_rows

            statements = batch[self.args.input_col].tolist()

            if self.use_prompt:
                if self.args.bias_induction_mode == 'two-step':
                    # Monolingual bias induction (step 1)
                    topics = batch[self.args.topic_col].tolist()
                    statements = self.transform_batch(statements, topics)
                    add_bias = False  # no need to re-add bias in translations
                    batch['transformation'] = statements

                translations = self.translate_batch_with_prompt(
                    statements,
                    self.system_instructions[add_bias],
                    src_lang=self.src_lang_name,
                    tgt_lang=self.tgt_lang_name,
                )
            else:
                src_id, tgt_id = self.tokenizer.convert_tokens_to_ids(
                    [self.args.src_lang, self.args.tgt_lang]
                )
                translations = self.translate_batch(statements, tgt_id)

            assert len(translations) == len(batch), "Translation batch size mismatch!"
            batch['translation'] = translations
            batch['target_lang'] = self.args.tgt_lang[:2].lower()

            if self.args.RTT:
                if self.use_prompt:
                    rtt = self.translate_batch_with_prompt(
                        translations,
                        self.system_instructions[False],  # no bias
                        src_lang=self.tgt_lang_name,
                        tgt_lang=self.src_lang_name,
                    )
                else:
                    rtt = self.translate_batch(translations, src_id)

                assert len(rtt) == len(batch), "rtt batch size mismatch!"
                batch['rtt'] = rtt

            # Append batch to output file
            batch[all_columns].to_csv(output_file, mode='a', header=False, index=False)

            rows_seen += batch_size

        print("Done!")


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.save_config:
        config_path = Path(args.save_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json_args = {k: v for k, v in vars(args).items()}
            json.dump(json_args, f, indent=2)
        print(f"Saved config to {config_path}")

    translator = TranslationPipeline(args)
    translator.run()


if __name__ == '__main__':
    main()