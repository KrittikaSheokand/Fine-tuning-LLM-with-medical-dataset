#Import required libraries
from dataclasses import dataclass, field
import os
from typing import Optional

#Define arguments for model training and output storage
@dataclass
class ScriptArguments:
    #Hugging face token
    hf_token: str = field(metadata={"help": ""})

    #Model to be used - Llama 2 7B here
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": ""}
    )

    seed: Optional[int] = field(
        default=4761, metadata = {'help':''}
    )
    #Path for data
    data_path: Optional[str] = field(
        default="./data/forums_short.json", metadata={"help": ""}
    )
    #Directory to store output in
    output_dir: Optional[str] = field(
        default="output", metadata={"help": ""}
    )
    #Training batch size    
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":""}
    )
    #Gradient accumulation steps
    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":""}
    )
    #Optimizer model
    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":""}
    )
    #Save steps
    save_steps: Optional[int] = field(
        default = 2000, metadata = {"help":""}
    )
    #Logging steps
    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":""}
    )
    #Learning rate
    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":""}
    )
    #Maximum grad norm
    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":""}
    )
    #Training epochs
    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":""}
    ) 

    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":""}
    )

    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":""}
    ) 
    #Output directory
    lora_dir: Optional[str] = field(default = "./model/llm_hate_speech_lora", metadata = {"help":""})
    #Max steps
    max_steps: Optional[int] = field(default=-1, metadata={"help": ""})
    #Chat sample as text field
    text_field: Optional[str] = field(default='chat_sample', metadata={"help": ""})


