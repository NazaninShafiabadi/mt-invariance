""" 
Sample usage:

python src/finetune.py \
--model="/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.2-3B" \
--tokenizer="/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.2-3B" \
--train_file="data/BiMultiSD/train.csv" \
--val_files="data/BiMultiSD/valid.csv" \
--output_dir="models/llama+bimultisd" \
--max_len=512 \
--topic_col="target" \
--text_col="comment" \
--label_col="label" \
--num_labels=2 \
--num_epochs=10 \
--batch_size=16 \
--save_config
"""

import os
import argparse
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
import shutil
import torch
import torch.multiprocessing as mp
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    logging,
    EarlyStoppingCallback,
)
from transformers.utils import is_flash_attn_2_available
from peft import LoraConfig, get_peft_model, TaskType
from preprocessing import InputPreprocessor

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
os.environ["WANDB_DISABLED"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, type=str, help="Model name or path")
    parser.add_argument("--tokenizer", default=None, type=str, help="Tokenizer name or path")
    parser.add_argument("--train_file", required=True, type=str, help="Path to training data")
    parser.add_argument("--val_files", required=True, nargs="+",
        help="Paths to one or more validation datasets"
    )
    parser.add_argument("--output_dir", default="finetuned_model", type=str, help="Output directory")
    parser.add_argument("--balance_by", default=None, nargs="+",
        help="List of attributes (column names) to balance by. Separate by space.",
    )
    parser.add_argument("--max_len", default=512, type=int, help="Maximum sequence length")
    parser.add_argument("--topic_col", default="target", type=str, help="Topic column name")
    parser.add_argument("--text_col", default="comment", type=str, help="Text column name")
    parser.add_argument("--label_col", default="label", type=str, help="Label column name")
    parser.add_argument("--num_labels", default=2, type=int, help="Number of labels")
    parser.add_argument("--num_epochs", default=3, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
    parser.add_argument("--metric_for_best_model", default="loss", type=str, choices=["loss", "accuracy", "f1"])
    parser.add_argument("--label_map", default=None, type=str, help="Path to label mapping file")
    parser.add_argument("--multi_eval", action="store_true",
        help="Evaluate on all validation sets; but best model will be chosen based on performance on the first set."
    )
    parser.add_argument("--save_checkpoints", action="store_true", help="Save checkpoints during training")
    parser.add_argument("--save_config", action="store_true", help="Save the run configuration as a JSON file.")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation")
    parser.add_argument("--eval_strategy", default="steps", type=str, choices=["steps", "epoch"])
    
    return parser


def ensure_padding_tokens(tokenizer, model):
    """
    Ensures that tokenizer and model have pad_token and pad_token_id set.
    If missing, assigns tokenizer.eos_token as pad_token.
    """
    if not tokenizer.pad_token and hasattr(tokenizer, "eos_token"):
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


def get_lora_model(model):
    """
    Wrap model with LoRA adapters. Automatically picks target_modules
    depending on the HuggingFace model type.
    """
    model_type = getattr(model.config, "model_type", "").lower()
    target_modules = None

    if model_type in ["llama", "gpt2", "falcon", "mistral"]:
        target_modules = ["q_proj", "v_proj"]  # decoder-only models
    elif model_type in ["bert", "roberta", "xlm-roberta", "electra"]:
        target_modules = ["query", "value"]  # encoder-only models

    # LoRA configuration for sequence classification task
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,  # Training mode
        r=8,  # Rank (controls how much LoRA modifies the model; smaller = more compression)
        lora_alpha=16,  # Scaling factor (higher = stronger adaptation)
        lora_dropout=0.1,  # dropout probability for LoRA layers (helps prevent overfitting)
        target_modules=target_modules,
    )
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    return model


def compute_metrics(pred):
    logits, labels = pred
    logits = torch.tensor(logits)
    probs = logits.softmax(dim=-1)
    confidences, preds = probs.max(dim=-1)

    # for i in range(min(3, len(labels))):
    #     print(f"> Pred: {preds[i]} | Confidence: {confidences[i].item()} | Label: {labels[i]}")

    # Compute metrics
    average = "binary" if len(set(labels)) == 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average, zero_division=0
    )
    acc = balanced_accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main(args):
    # Load model and tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.num_labels,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # **({"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {})
        attn_implementation="sdpa",
    )

    print(f"Using device: {model.device}")

    ensure_padding_tokens(tokenizer, model)

    if args.use_lora:
        print("Applying LoRA...")
        model = get_lora_model(model)  # Apply LoRA to the model
        model.print_trainable_parameters()

    preprocessor = InputPreprocessor(
        tokenizer, max_len=args.max_len, device=model.device
    )

    if args.label_map:
        if os.path.exists(args.label_map):
            preprocessor.load_label_map(args.label_map)
        else:
            print(f"No such file or directory: {args.label_map}")
            print("This path will be ignored and a new label map will be created.")

    trainset = preprocessor.process(
        args.train_file,
        args.topic_col,
        args.text_col,
        args.label_col,
        balance_by=args.balance_by,
    )
    validsets = {
        os.path.basename(val_file): preprocessor.process(
            val_file, args.topic_col, args.text_col, args.label_col
        )
        for val_file in args.val_files
    }

    # Save label map for consistency during inference
    preprocessor.save_label_map(
        os.path.join(args.output_dir, "label_map.json")
    )

    first_valset_name, first_valset = list(validsets.items())[0]
    metric = args.metric_for_best_model
    if args.multi_eval:
        eval_dataset = validsets
        metric_for_best_model = f"eval_{first_valset_name}_{metric}"
    else:
        eval_dataset = first_valset
        metric_for_best_model = f"eval_{metric}"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_strategy=args.eval_strategy,
        logging_steps=500,
        logging_first_step=True,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Use 10% of training steps for warm-up
        max_grad_norm=1.0,  # 5.0
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,  # Default weight decay: 0.01
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        bf16=torch.cuda.is_bf16_supported(),  # mixed-precision training on H100, A100, or Apple M-series
        fp16=True if not torch.cuda.is_bf16_supported() else False,
        dataloader_num_workers=4,
        report_to=None,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        # default optimizer: AdamW
    )

    # Training
    print("Training...")
    trainer.train()

    print("Evaluating...")
    for val_name, dataset in validsets.items():
        print(f"Evaluating on {val_name}...")
        results = trainer.evaluate(eval_dataset=dataset)
        # print(f"Results for {val_name}: {results}")

    best_ckpt = trainer.state.best_model_checkpoint
    print(f"Best checkpoint selected: {best_ckpt}")

    # Copy the trainer_state from the best checkpoint to the best model directory
    best_model = os.path.join(args.output_dir, "best_ckpt")
    os.makedirs(best_model, exist_ok=True)
    shutil.copy(
        os.path.join(best_ckpt, "trainer_state.json"),
        os.path.join(best_model, "trainer_state.json"),
    )

    if args.use_lora:
        from peft import PeftModel

        # Load clean base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=args.num_labels
        )
        # Attach trained LoRA adapter from the best checkpoint
        model = PeftModel.from_pretrained(base_model, best_ckpt)
        # model = get_lora_model(model)

        # Save adapters only (lightweight)
        adapter_dir = os.path.join(args.output_dir, "lora_adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"LoRA adapters saved to {adapter_dir}")

        # Merge adapters into the base model for standalone inference
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(best_model)
        tokenizer.save_pretrained(best_model)
        print(f"Merged model saved to {best_model}")
    else:
        trainer.save_model(best_model)
        tokenizer.save_pretrained(best_model)
        print(f"Model saved to {best_model}")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # safety measure to avoid fork-related CUDA bugs

    # print(f'Using device: {DEVICE}')

    parser = create_parser()
    args = parser.parse_args()

    if args.save_config:
        config_path = os.path.join(args.output_dir, "training_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json_args = {k: v for k, v in vars(args).items()}
            import json

            json.dump(json_args, f, indent=2)
        print(f"Saved config to {config_path}")

    main(args)
