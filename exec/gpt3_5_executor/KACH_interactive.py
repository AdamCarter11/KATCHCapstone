# pip install openai
import json
import os
import sys
import numpy as np
import openai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import logging
from transformers import logging as hf_logging
import openai
hf_logging.set_verbosity_error()

# Set up OpenAI API key
openai.api_key = "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"

# Load BERT tokenizer and model for embeddings
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

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




def find_video_tags_and_transcript(prompt, video_metadata):
    words = prompt.lower().split()
    for video_key, video_info in video_metadata.items():
        # Check video transcript portions
        for segment in video_info["transcript_portions"]:
            start_time = int(segment["start_time"])
            end_time = int(segment["end_time"])
            for tag in segment["tags"]:
                if tag["word"].lower() in words and start_time <= int(tag["timestamp"]) <= end_time:
                    return tag["word"], segment["text"], tag["timestamp"], segment["start_time"], segment["end_time"]
    return "", "", "", "", ""







def interact_with_gpt3_5(prompt, video_metadata):
    model_engine = "text-davinci-003"

    # Find the video tags and transcript segment that match the prompt
    video_tags, transcript_segment, _, _, _ = find_video_tags_and_transcript(prompt, video_metadata)

    # Create a context from the tags and transcript
    context = f"In a video tagged with {video_tags}, a segment of the transcript reads: \"{transcript_segment}\". "

    # Append the user's prompt to the context
    full_prompt = context + prompt

    # Generate a response from GPT-3.5
    response = openai.Completion.create(
        engine=model_engine,
        prompt=full_prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()



# Define the full path to your JSON file
file_path = 'C:\\Users\\jpiye\\OneDrive\\Documents\\GitHub\\KATCHCapstone\\exec\\gpt3_5_executor\\response.json'

# Now you can open the file with this path
with open(file_path) as f:
    video_metadata = json.load(f)

if __name__ == "__main__":
    stored_tags = ""
    while True:
        user_input = input("Enter a prompt for KACH (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = interact_with_gpt3_5(user_input, video_metadata)
        print("KACH response:", response)