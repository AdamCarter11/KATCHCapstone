# pip install openai
import json
import openai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

from transformers import logging as hf_logging

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

import openai

def find_video_tags_and_transcript(prompt, video_metadata):
    words = prompt.lower().split()
    for video_key, video_info in video_metadata.items():
        # Check video tags
        for tag in video_info["video_tags"]:
            if tag.lower() in words:
                timestamps = video_info["tag_timestamps"].get(tag, [])
                for segment in video_info["transcript_segments"]:
                    if segment["timestamp"] in timestamps:
                        return ', '.join(video_info["video_tags"]), segment["transcript"]
    return "", ""

def interact_with_gpt3_5(prompt, video_metadata):
    model_engine = "text-davinci-003"

    # Find the video tags and transcript segment that match the prompt
    video_tags, transcript_segment = find_video_tags_and_transcript(prompt, video_metadata)

    # Create a context from the tags
    context = f"In a video tagged with {video_tags}, a segment of the transcript reads: \"{transcript_segment}\". "

    # Append the user's prompt to the context
    context += prompt

    # Generate a response from GPT-3.5
    response = openai.Completion.create(
        engine=model_engine,
        prompt=context,
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
file_path = 'C:\\Users\\jpiye\\OneDrive\\Documents\\GitHub\\KATCHCapstone\\exec\\gpt_executor\\video_metadata.json'

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
