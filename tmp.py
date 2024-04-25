from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")

model.save_pretrained("Qwen/Qwen1.5-0.5B")
tokenizer.save_pretrained("Qwen/Qwen1.5-0.5B")