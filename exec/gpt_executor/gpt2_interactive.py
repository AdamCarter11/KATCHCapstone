from transformers import GPT2Tokenizer, GPT2LMHeadModel

def interact_with_gpt2(prompt):
    model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    while True:
        user_input = input("Enter a prompt for GPT-2: ")
        response = interact_with_gpt2(user_input)
        print("GPT-2 Response:", response)
