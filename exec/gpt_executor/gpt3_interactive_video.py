#pip install openai
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

# Set up OpenAI API key
openai.api_key = "your_api_key_here"

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

def interact_with_gpt3_5(prompt, video_metadata, stored_transcript):
    model_engine = "text-davinci-003"

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

    # Split transcript into chunks of 2048 tokens
    transcript_chunks = [transcript[i:i + 2048] for i in range(0, len(transcript), 2048)]

    # Send transcript chunks one by one
    for chunk in transcript_chunks:
        context = f"In a video about {video_tags}, the speaker talks about \"{chunk}\", please do not generate a " \
                  f"response about it unless I specifically ask about the contents."

        openai.Completion.create(
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

    # Send the user's prompt with the original context statement

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
        user_input = input("Enter a prompt for KACH (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = interact_with_gpt35(user_input, video_metadata, stored_transcript)
        print("GPT-3.5 Response:", response)