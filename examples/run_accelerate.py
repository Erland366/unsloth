import os
import torch
import wandb
import torch.distributed as dist
import accelerate
import textwrap

from unsloth import FastLanguageModel
from transformers import set_seed
from unsloth import FastModel
from dotenv import load_dotenv

from trl import SFTTrainer, SFTConfig, get_kbit_device_map
from unsloth import is_bfloat16_supported

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))

# os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/aios-stor/erland/.hf_cache"
set_seed(3407)

WANDB_PROJECT_NAME = "unsloth_experiment"
RUN_NAME = "Qwen3_30B_A3B"

os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
os.environ["WANDB_MODE"] = "online"
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"
# If you want to debug
# os.environ["UNSLOTH_COMPILE_DEBUG"] = "1"         
# os.environ["UNSLOTH_COMPILE_MAXIMUM"] = "1"         
# os.environ["UNSLOTH_COMPILE_IGNORE_ERRORS"] = "1"         

def main():
    max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

        "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth

    qwen_models = [
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",      # Qwen 14B 2x faster
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth

    is_accelerate = "ACCELERATE_" in "".join(os.environ.keys())
    print(f"{is_accelerate = }")

    # if accelerator.is_main_process:
    #     print(f"Running with {accelerator.num_processes} processes")
    #     import pdb; pdb.set_trace()

    # accelerator.wait_for_everyone()

    # import pdb; pdb.set_trace(); # Use this to debug the model loading

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-0.6B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        use_exact_model_name = True, 
        device_map=get_kbit_device_map(),
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    current_rank = int(os.environ.get("RANK", "0"))


    model = FastLanguageModel.get_peft_model(
        model,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj"],
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # first_param_name, first_param = next(model.named_parameters())
    # # Focus on a LoRA parameter if possible, or just any parameter
    # # LoRA parameters are typically like 'base_model.model....lora_A.weight'
    # lora_params_found = False
    # for name, param in model.named_parameters():
    #     print(f"{name = }, {param = }", f"{os.environ['RANK'] = }")
    # for name, param in model.named_parameters():
    #     if "lora" in name.lower():
    #         print(f"[Rank {current_rank}] First LoRA parameter ('{name}') properties: dtype={param.dtype}, device={param.device}, shape={param.shape}, requires_grad={param.requires_grad}")
    #         lora_params_found = True
    #         break

    alpaca_prompt = textwrap.dedent("""\
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
    )

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    from datasets import load_dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps=5,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )

    # for name, param in trainer.model.named_parameters():
    #     print(f"{name = }, {param = }", f"{os.environ['RANK'] = }")

    trainer_stats = trainer.train()
    print(f"{trainer_stats = }")

if __name__ == "__main__":
    main()