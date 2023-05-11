import json
import os
import sys
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
from transformers import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logging.set_verbosity_error()
#auto-gpt open source tool that uses gpt to create autonomous AI tool, 3.5 is better than 4 for indexing into text

def get_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


def get_keywords(text):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)

    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]

    if not filtered_words:
        return set()

    X = vectorizer.fit_transform([' '.join(filtered_words)])
    feature_names = vectorizer.get_feature_names_out()
    return set(feature_names)


def interact_with_gpt2(prompt, video_metadata, stored_transcript):
    model_name = "gpt2-large"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    video_key = None
    words = prompt.split()
    for word in words:
        if word.isdigit():
            video_key = "video_" + word
            break

    if video_key and video_key in video_metadata:
        transcript = video_metadata[video_key]["transcript"]
        stored_transcript = transcript
        video_tags = ', '.join(video_metadata[video_key]["video_tags"])
    else:
        transcript = stored_transcript
        video_tags = ""

    user_keywords = get_keywords(prompt)
    transcript_keywords = get_keywords(transcript)

    user_input_embedding = get_embedding(prompt)
    transcript_embedding = get_embedding(transcript)
    similarity_score = cosine_similarity(user_input_embedding, transcript_embedding)

    if len(user_keywords.intersection(transcript_keywords)) > 0 or similarity_score > 0.4:
        context = f"In a video about {video_tags}, the speaker talks about \"{transcript}\". {prompt}"
    else:
        context = prompt

    inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=150)
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=10,  # Generate 10 responses
                             do_sample=True, temperature=0.7, top_k=50, top_p=0.85, no_repeat_ngram_size=4)

    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Rank the responses based on their similarity to the user's input
    response_embeddings = [get_embedding(response.replace(context, "").strip()) for response in responses]
    similarities = [cosine_similarity(user_input_embedding, response_embedding) for response_embedding in
                    response_embeddings]
    best_response_index = np.argmax(similarities)

    response = responses[best_response_index].replace(context, "").strip()
    return response

# Load BERT tokenizer and model for embeddings
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

import nltk
from nltk.corpus import stopwords
# Comment out the following lines
# nltk.download("stopwords")
# nltk.download("punkt")

# Load JSON metadata
json_string = '''
{
  "video_1": {
    "transcript": "In this video, we discuss the process of photosynthesis. Jeremy is a Cat",
    "video_tags": ["Jeremy", "biology", "plants", "science"],
    "timestamps": [0, 15, 30, 45]
  }
}
'''

video_metadata = json.loads(json_string)

if __name__ == "__main__":
    stored_transcript = ""
    while True:
        user_input = input("Enter a prompt for GPT-2 (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = interact_with_gpt2(user_input, video_metadata, stored_transcript)
        print("GPT-2 Response:", response)
