import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"
MODEL = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"

print(f"Loading: {MODEL}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.eval()

print("Model loaded.")
print(f"GPU: {torch.cuda.get_device_name(0)}")

while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        outputs = model.generate(
            **inputs,
            max_new_tokens=512*4,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
        )

        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event) / 1000

    generated_tokens = outputs.shape[1] - input_tokens
    tok_per_sec = generated_tokens / elapsed

    response = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True,
    )

    print("\nQwen:")
    print(response)

    print(
        f"\nGenerated: {generated_tokens} tokens"
        f"\nTime:      {elapsed:.2f}s"
        f"\nSpeed:     {tok_per_sec:.2f} tok/s"
    )
