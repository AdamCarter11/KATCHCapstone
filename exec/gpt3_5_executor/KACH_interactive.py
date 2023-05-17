import json
import numpy as np
import openai
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Set up OpenAI API key
openai.api_key = 

# Load BERT tokenizer and model for embeddings
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, :, :].detach().numpy()
    sentence_embedding = np.mean(embeddings, axis=1)
    return sentence_embedding[0]

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

from scipy.spatial.distance import cosine

def find_video_tags_and_transcript(prompt, video_metadata):
    words = get_keywords(prompt)
    prompt_embedding = get_embedding(prompt)
    best_match = ("", "", "", "", "", -1)  # (tag, transcript_text, tag_timestamp, start_time, end_time, similarity)
    prompt_embedding_2d = np.reshape(prompt_embedding, (1, -1))

    for video in video_metadata["videos"]:
        for segment in video["transcript_portions"]:
            start_time = int(segment["start_time"])
            end_time = int(segment["end_time"])
            transcript_embedding = get_embedding(segment["text"])
            transcript_embedding_2d = np.reshape(transcript_embedding, (1, -1))
            similarity = cosine_similarity(prompt_embedding_2d, transcript_embedding_2d)[0][0]

            if similarity > best_match[5]:
                for tag in segment["tags"]:
                    if tag["word"].lower() in words:
                        best_match = (
                        tag["word"], segment["text"], tag["timestamp"], segment["start_time"], segment["end_time"],
                        similarity)

    return best_match[0], best_match[1], best_match[2], best_match[3], best_match[4]

def interact_with_gpt3_5(prompt, video_metadata):
    model_engine = "text-davinci-003"

    # Find the video tags and transcript segment that match the prompt
    video_tags, transcript_segment, _, _, _ = find_video_tags_and_transcript(prompt, video_metadata)

    # Create a context from the tags and transcript
    context = f"In a video tagged with {video_tags}, a segment of the transcript reads: \"{transcript_segment}\". "

    # Append the user's prompt to the context
    full_prompt = context + "What can you teach me about it?"

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
