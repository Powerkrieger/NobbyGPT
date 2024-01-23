import os
import torch
import wandb
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from datetime import datetime


base_model_name = "Llama-2-7b-hf"
base_model_id = f"meta-llama/{base_model_name}"
train_data_file = '../../Data/Training_Data/WeeveIE_Wikipedia/WeeveLVL3_J.json'
eval_data_file = '../../Data/Training_Data/WeeveIE_Wikipedia/WeeveLVL3_J.json'
project_name = "denglish-weeve_lvl3"


def formatting_func(example):
    text = f"{example['input']}"
    return text


def load_model(base_model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
    return model


def load_tokenizer(base_model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_datasets(tokenizer, train_dataset, eval_dataset, max_length=256):
    def generate_and_tokenize_prompt(prompt):
        result = tokenizer(
            formatting_func(prompt),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    return tokenized_train_dataset, tokenized_val_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def wandb_setup(wandb_project):
    wandb.login()
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project


def lora_setup(model_for_lora):
    model_for_lora.gradient_checkpointing_enable()  # requires setting reentrant explicitly
    model_for_lora = prepare_model_for_kbit_training(model_for_lora)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model_for_lora = get_peft_model(model_for_lora, config)
    print_trainable_parameters(model_for_lora)

    # Apply the accelerator. You can comment this out to remove the accelerator.
    model_for_lora = accelerator.prepare_model(model_for_lora)
    return model_for_lora


def evaluate_model(model, tokenizer):
    # prepare model input
    eval_prompt = "Write a story about a sailor. "  # End of sentence, we just want to see what is the output
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    # test evaluate model
    model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0],
                               skip_special_tokens=True))


def setup_accelerator():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    return accelerator


def setup_finetuning(model, tokenizer, project, base_model_name,
                     tokenized_train_dataset, tokenized_eval_dataset):
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        print("more than 1 ...")
        model.is_parallelizable = True
        model.model_parallel = True

    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            max_steps=500,
            learning_rate=2.5e-5,  # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=50,  # Save checkpoints every 50 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=50,  # Evaluate and save checkpoints every 50 steps
            do_eval=True,  # Perform evaluation at the end of training
            report_to="wandb",  # Comment this out if you don't want to use weights & baises
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"  # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    return trainer


if __name__ == '__main__':
    # set up wandb
    wandb_setup(project_name)

    # Accelerator
    accelerator = setup_accelerator()

    # load dataset
    train_dataset = load_dataset('json', data_files=train_data_file,
                                 split='train')
    eval_dataset = load_dataset('json', data_files=eval_data_file,
                                split='train')

    # prepare model and tokenizer
    model = load_model(base_model_id)
    tokenizer = load_tokenizer(base_model_id)
    tokenized_train_dataset, tokenized_eval_dataset = tokenize_datasets(tokenizer, train_dataset, eval_dataset)
    print(tokenized_train_dataset[1]['input_ids'])

    # evaluate unfitted model
    evaluate_model(model, tokenizer)

    # set up lora
    model = lora_setup(model)

    # finetune
    trainer = setup_finetuning(model, tokenizer, project_name, base_model_name,
                               tokenized_train_dataset, tokenized_eval_dataset)
    trainer.train()
