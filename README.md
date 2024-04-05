# Fine-tuning-LLM-with-medical-dataset
Fine tuned Llama 2 7B with medical dataset using QLoRA 

I used medical dataset to fine tune Llama-2-7B model so that I can get medical data related answers

LoRA is dividing weights into two new low rank matrices which are trained and then added to original weights. This is done because training on original weights would be tough on computation as they are huge
QLoRA is quantizing data (reduces to 4 bit or 8 bit precision) before LoRa which reduces memory used but still gives good results.

Instruction fine-tuning comes after the unsupervised pre-training. Unsupervised pre-training trains LLM on huge data so that it can predict the next token given previous tokens. However, instruction fine-tuning is making it more like a chatbot which can give answers to questions. We are instruction fine tuning here as in the data we have human and AI comments.

Huggingface Directory of output: kkrittik/medical_hw_test
Medical dataset: https://huggingface.co/datasets/jbrophy123/medical_dataset
