# from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# import torch

# MODEL_PATH = "./llama3.2-1B"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "write me a http code in c"}
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# inputs = tokenizer(text, return_tensors="pt").to(model.device)

# # --- Option 1: Normal (wait for full response) ---
# # outputs = model.generate(
# #     **inputs,
# #     max_new_tokens=512,
# #     temperature=0.7,
# #     do_sample=True,
# #     pad_token_id=tokenizer.eos_token_id
# # )
# # response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
# # print(response)

# # --- Option 2: Streaming (uncomment to use) ---
# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
# model.generate(
#     **inputs,
#     max_new_tokens=2048,
#     temperature=0.7,
#     do_sample=True,
#     pad_token_id=tokenizer.eos_token_id,
#     streamer=streamer
# )

# from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# import torch
# import time

# MODEL_PATH = "./llama3.2-1B"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH,
#     dtype=torch.bfloat16,
#     device_map="auto"
# )

# messages = [
#     {"role": "system", "content": "you are a hacker with good profeciency in c"},
#     {"role": "user", "content": "give me a assembly code that switches context and also a c struct that stores the context"}
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# inputs = tokenizer(text, return_tensors="pt").to(model.device)

# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# print("\n--- Response ---")
# start_time = time.perf_counter()

# output = model.generate(
#     **inputs,
#     max_new_tokens=32000,
#     temperature=0.6,
#     do_sample=True,
#     pad_token_id=tokenizer.eos_token_id,
#     streamer=streamer
# )

# end_time = time.perf_counter()

# # Calculate stats
# input_tokens = inputs.input_ids.shape[-1]
# output_tokens = output.shape[-1] - input_tokens
# elapsed = end_time - start_time
# tokens_per_sec = output_tokens / elapsed

# print("\n--- Stats ---")
# print(f"  Input tokens   : {input_tokens}")
# print(f"  Output tokens  : {output_tokens}")
# print(f"  Time elapsed   : {elapsed:.2f}s")
# print(f"  Speed          : {tokens_per_sec:.1f} tokens/sec")

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
import torch
import time

MODEL_PATH = "./llama3.2-1B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # quantizes the scale factors too, saves a bit more VRAM
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa"
)
print("Model loaded! Type 'exit' or 'quit' to stop.\n")

conversation = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Bye!")
        break

    conversation.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\nAssistant: ", end="", flush=True)
    start_time = time.perf_counter()

    output = model.generate(
        **inputs,
        max_new_tokens=32000,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    end_time = time.perf_counter()

    input_tokens = inputs.input_ids.shape[-1]
    output_tokens = output.shape[-1] - input_tokens
    elapsed = end_time - start_time
    tokens_per_sec = output_tokens / elapsed

    assistant_reply = tokenizer.decode(output[0][input_tokens:], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": assistant_reply})

    print(f"\n[{output_tokens} tokens | {elapsed:.2f}s | {tokens_per_sec:.1f} tok/s]\n")