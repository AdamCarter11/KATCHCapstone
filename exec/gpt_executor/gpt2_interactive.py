from transformers import GPT2Tokenizer, GPT2LMHeadModel

def interact_with_gpt2(prompt, context=''):
    model_name = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad token to the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    if context:
        prompt_with_context = f"Given the following video metadata:\n{context}\n{prompt}"
    else:
        prompt_with_context = prompt

    inputs = tokenizer(prompt_with_context, return_tensors="pt", padding=True, truncation=True, max_length=150)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if context:
        return generated_text.strip(prompt_with_context)  # Remove the context and prompt from the response
    else:
        return generated_text

def should_use_context(user_input):
    context_keywords = ['transcript', 'video tags', 'timestamps']
    return any(keyword.lower() in user_input.lower() for keyword in context_keywords)

if __name__ == "__main__":
    video_metadata = """
    Transcript: Sample transcript text.
    Video Tags: tag1, tag2, tag3
    Timestamps: 1, 2, 3
    """

    while True:
        user_input = input("Enter a prompt for GPT-2: ")

        if should_use_context(user_input):
            response = interact_with_gpt2(user_input, video_metadata)
        else:
            response = interact_with_gpt2(user_input)

        print("GPT-2 Response:", response.strip())
