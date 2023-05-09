from transformers import GPT2Tokenizer, GPT2LMHeadModel

def interact_with_gpt2(prompt):
    model_name = "gpt2-medium"  # You can use "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad token to the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=150)
    outputs = model.generate(**inputs,
                             max_length=150,
                             num_return_sequences=1,
                             temperature=100,
                             top_k=10,
                             top_p=0.95,
                             no_repeat_ngram_size=3)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

if __name__ == "__main__":
    while True:
        user_input = input("Enter a prompt for GPT-2: ")
        response = interact_with_gpt2(user_input)
        print("GPT-2 Response:", response)
