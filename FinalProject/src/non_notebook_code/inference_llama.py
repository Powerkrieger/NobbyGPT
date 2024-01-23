import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from finetune_llama import evaluate_model

if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model_name = "Llama-2-7b-hf"
    base_model_id = f"meta-llama/{base_model_name}"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Llama 2 7B, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

    ft_model = PeftModel.from_pretrained(base_model, "llama2-7b-whole-text-denglish-finetune/checkpoint-500")

    evaluate_model(ft_model, tokenizer)