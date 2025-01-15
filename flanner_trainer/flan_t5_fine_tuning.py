import argparse
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
)
from datasets import load_from_disk
import torch
import nltk
from model_config import *

nltk.download("punkt", quiet=True)


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="Path to the already processed dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=BATCH_SIZE, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=BATCH_SIZE, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=GENERATION_MAX_LEN, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=GENERATION_NUM_BEAMS, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=DEEP_SPEED_CONFIG_PATH, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Perform gradient checkpointing to save memory on account of ~20% of the speed")
    parser.add_argument("--output_dir", type=str, default=MODEL_PATH, help="Path to the local fined tuned model.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args


def training_function(args):
    
    torch.cuda.empty_cache()
    print(args)
    set_seed(args.seed)

    output_dir = args.output_dir
    
    # load dataset from disk
    train_dataset = load_from_disk(args.dataset_path)
    # train_dataset = load_from_disk(f'{args.dataset_path}_train')
    # eval_dataset = load_from_disk(f'{args.dataset_path}_validation')
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=LABEL_PAD_TOKEN_ID, pad_to_multiple_of=8
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        # fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        
        # logging & evaluation strategies
        # evaluation_strategy="steps",  # "epoch",
        logging_strategy="steps",
        logging_steps=500,
        logging_dir=f"{args.output_dir}/logs",
        save_strategy="epoch",
        save_total_limit=EPOCHS,  # limit amount of checkpoints to save
        #load_best_model_at_end=True,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
    
    # Start training
    trainer.train(
        resume_from_checkpoint=RESUME_CHECKPOINT_PATH
    )

    tokenizer.save_pretrained(args.output_dir)


def main():
    args, _ = parse_args()
    training_function(args)


if __name__ == "__main__":
    main()
