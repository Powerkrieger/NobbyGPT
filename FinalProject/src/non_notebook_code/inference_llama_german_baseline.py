import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import nobbygpt_config

#base_model_name = nobbygpt_config.base_model_name
#base_model_id = nobbygpt_config.base_model_id
base_model_name = "Llama-2-7b-german-assistant-v3"
base_model_id = f"flozi00/{base_model_name}"
# project_name = nobbygpt_config.project_name
# project_name = "denglish-weeve_lvl4"


def evaluate_model(model, tokenizer, max_tokens=256):
    # prepare model input
    eval_prompt = "Write a story about a sailor. "  # End of sentence, we just want to see what is the output
    eval_prompt = 'Bitte generiere einen langen Aufsatz, basierend auf dem folgenden Anfang: "Hallo, mein Name ist".'
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    # test evaluate model
    # llama-2 was trained wit a context length of 4096, so it should not be out of context to generate so many tokens
    # finetuning was only with sample length of 256, more seems to generate dumbness
    model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=max_tokens, pad_token_id=2)[0],
                               skip_special_tokens=True))


if __name__ == "__main__":
    project_name = sys.argv[1]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Llama 2 7B, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True,
        token=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    run_name = base_model_name + "-" + project_name
    ft_model = PeftModel.from_pretrained(base_model, f"{run_name}/checkpoint-500")
    tf_model = base_model

    for i in range(100):
        evaluate_model(ft_model, tokenizer)
