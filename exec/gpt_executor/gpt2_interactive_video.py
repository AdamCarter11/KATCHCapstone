import json
import os
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
import torch
from transformers import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.set_verbosity_error()

def interact_with_gpt2(prompt, video_metadata, stored_transcript):
    model_name = "gpt2-medium"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad token to the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    # Load BERT tokenizer and model for embeddings
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    import nltk
    from nltk.corpus import stopwords
    # Comment out the following lines
    # nltk.download("stopwords")
    # nltk.download("punkt")

    def get_embedding(text):
            inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = bert_model(**inputs)
            return outputs.last_hidden_state[:, 0, :].detach().numpy()

    video_key = None
    words = prompt.split()
    for word in words:
        if word.isdigit():
            video_key = "video_" + word
            break

    if video_key and video_key in video_metadata:
        transcript = video_metadata[video_key]["transcript"]
        stored_transcript = transcript
    else:
        transcript = stored_transcript

    def get_keywords(text):
        words = nltk.word_tokenize(text)
        words = [word.lower() for word in words if word.isalnum()]
        stop_words = set(stopwords.words("english"))
        keywords = [word for word in words if word not in stop_words]
        return set(keywords)

    user_keywords = get_keywords(prompt)
    transcript_keywords = get_keywords(transcript)

    # Compute the similarity between user input and transcript
    user_input_embedding = get_embedding(prompt)
    transcript_embedding = get_embedding(transcript)
    similarity_score = cosine_similarity(user_input_embedding, transcript_embedding)

    if len(user_keywords.intersection(transcript_keywords)) > 0 or similarity_score > 0.5:
        context = f"Transcript: \"{transcript}\"\n{prompt}"
    else:
        context = prompt

    inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1,
                             temperature=1.5, top_k=10, top_p=0.95, no_repeat_ngram_size=4)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = generated_text.replace(context, "").strip()
    return response

# Load JSON metadata
json_string = '''
{
  "video_1": {
    "transcript": "In this video, we discuss the process of photosynthesis, which is essential for plant growth.",
    "video_tags": ["photosynthesis", "biology", "plants", "science"],
    "timestamps": [0, 15, 30, 45]
  }
}
'''

video_metadata = json.loads(json_string)

if __name__ == "__main__":
    stored_transcript = ""
    while True:
        user_input = input("Enter a prompt for GPT-2: ")
        response = interact_with_gpt2(user_input, video_metadata, stored_transcript)
        print("GPT-2 Response:", response)
