import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_path = "./qwen3.6-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "user", "content": "i need your help i want to understand context switch in assembly and then port it to c so yea i need so detail"}
]

# Prepare input
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# Streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Generation kwargs
gen_kwargs = dict(
    **inputs,
    max_new_tokens=32000,
    streamer=streamer
)

# Run generation in separate thread (required for streaming)
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

# Measure time + tokens
start_time = time.time()
generated_tokens = 0

print("Output:\n", end="", flush=True)

for new_text in streamer:
    print(new_text, end="", flush=True)
    generated_tokens += len(tokenizer.encode(new_text, add_special_tokens=False))

end_time = time.time()

# Stats
elapsed = end_time - start_time
tps = generated_tokens / elapsed if elapsed > 0 else 0

print("\n\n--- Stats ---")
print(f"Tokens generated: {generated_tokens}")
print(f"Time taken: {elapsed:.2f} sec")
print(f"Tokens/sec: {tps:.2f}")