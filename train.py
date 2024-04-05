#Import required libraries
import os
import pandas

import transformers
from transformers import (
    AutoModelForCausalLM,#Get models
    AutoTokenizer,#To tokenize data
    set_seed,#To get same results
    BitsAndBytesConfig,#For quantization
    Trainer,#provides API for feature complete training in PyTorch, goes hand in hand with TrainingArguments class
    TrainingArguments,#To define arguments on which model is trained on 
    HfArgumentParser
)
from datasets import load_dataset
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

from trl import SFTTrainer

from utils import print_trainable_parameters, find_all_linear_names

from train_args import ScriptArguments

#Parameter-Efficient Fine-Tuning (PEFT)
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

#To parse the content defined in train_args
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

def training_function(args):
    
    #login using hugging face token
    login(token=args.hf_token)

    #Set seed to get same results
    set_seed(args.seed)

    #Get data path from train_args
    data_path=args.data_path

    #Load the data
    dataset = load_dataset(data_path)

    #Quanitizing the data 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # quantized to 4 bitn
        bnb_4bit_use_double_quant=True,#doble quantization to save additional 0.4bit per parameter
        bnb_4bit_quant_type="nf4",#Normalized float 4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,# 16 bits chosen for computatio
    )

    #Load pre-trained transformer model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        use_cache=False,
        device_map="auto", #automatically infer device maps
        quantization_config=bnb_config,  #use above defined bits nad bytes configuration
        trust_remote_code=True
    )

    #Get tokenizer from pre trained model 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token=tokenizer.eos_token # set padding as end of sentence
    tokenizer.padding_side='right' #padding to be applied on right side

    #Prepare the model for training by wraping the protocol
    model=prepare_model_for_kbit_training(model)

    #Get modules to apply LoRA to
    modules=find_all_linear_names(model)

    #Define configuration of LoRA model
    config = LoraConfig(
        r=64,#LoRA attention dimension
        lora_alpha=16, #Alpha parameter for LoRA scaling
        lora_dropout=0.1,#Dropout probability for LoRA layers
        bias='none',#No bias parameters to be trained
        task_type='CAUSAL_LM',
        target_modules=modules #Names of modules to apply LoRA to
    )

    #Adapt the model according to LoRA configurations defined above
    model=get_peft_model(model, config)
    #Define output directory
    output_dir = args.output_dir

    #Define and store training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir, #output directory to store model predictions
        per_device_train_batch_size=args.per_device_train_batch_size, # Batch size per GPU 
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Number of update steps to accumulate the gradients for
        optim=args.optim,#Optimizer to be used for training
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=False,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        tf32=False,
        report_to="none",
        push_to_hub=False,
        max_steps = args.max_steps
    )



    #Supervised fine tuning trainer to fine-tune the model
    trainer = SFTTrainer(
        model=model, #Model to be fine-tuned
        train_dataset=dataset['train'].select(range(2000)), #Training dataset
        dataset_text_field=args.text_field,
        max_seq_length=2048, #2048 set as maximum sequence length
        tokenizer=tokenizer, #use tokenizer defined above
        args=training_arguments #use training arguments defined above
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print('starting training')

    #Run SFT trainer
    trainer.train()

    print('LoRA training complete')
    #Save output
    lora_dir = args.lora_dir
    trainer.model.push_to_hub(lora_dir, safe_serialization=False)
    
    print("saved lora adapters")

    

if __name__=='__main__':
    training_function(args)

