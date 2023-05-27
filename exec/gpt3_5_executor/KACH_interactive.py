import json
import openai
import spacy
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# Set up OpenAI API key
openai.api_key = ""

# Load BERT tokenizer and model for embeddings
bert_tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
bert_model = BertModel.from_pretrained('deepset/bert-base-cased-squad2')

# Load Spacy model for keyword extraction
nlp = spacy.load("en_core_web_sm")


def get_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, :, :].detach().numpy()
    sentence_embedding = np.mean(embeddings, axis=1)
    return sentence_embedding[0]


def get_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == 'NOUN' or token.pos_ == 'PROPN']


def get_timestamp(keyword, video_metadata):
    for video in video_metadata["videos"]:
        for segment in video["transcript_portions"]:
            for tag in segment["tags"]:
                if tag["word"].lower() == keyword.lower():
                    return tag["timestamp"]
    return None


def convert_seconds_to_time_format(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}:{int(seconds):02d}"


SIMILARITY_THRESHOLD = 0.3
HIGH_SIMILARITY_THRESHOLD = 0.7


def find_video_tags_and_transcript(prompt, video_metadata):
    words = get_keywords(prompt)
    prompt_embedding = get_embedding(prompt)
    best_match = ("", "", "", "", "", "", -1)

    for video in video_metadata["videos"]:
        full_transcript = ' '.join([seg["text"] for seg in video["transcript_portions"]])
        transcript_embedding = get_embedding(full_transcript)
        similarity = cosine_similarity([prompt_embedding], [transcript_embedding])[0][0]

        if similarity > best_match[6]:
            matching_tags = [tag["word"] for seg in video["transcript_portions"] for tag in seg["tags"] if
                             tag["word"].lower() in words]
            best_match = (video["title"], matching_tags, full_transcript,
                          [get_timestamp(word, video_metadata) for word in matching_tags],
                          video["transcript_portions"][0]["start_time"], video["transcript_portions"][-1]["end_time"],
                          similarity)

    start_time_formatted = convert_seconds_to_time_format(float(best_match[4]))
    end_time_formatted = convert_seconds_to_time_format(float(best_match[5]))

    return best_match[0], best_match[1], best_match[2], best_match[3], start_time_formatted, end_time_formatted, \
    best_match[6]


def interact_with_gpt3_5(prompt, video_metadata):
    model_engine = "text-davinci-003"

    video_title, video_tags, transcript_segment, _, start_time_formatted, end_time_formatted, similarity = find_video_tags_and_transcript(
        prompt, video_metadata)

    if similarity > HIGH_SIMILARITY_THRESHOLD and (video_tags or transcript_segment):
        context = f"In the video titled '{video_title}', from {start_time_formatted} to {end_time_formatted}, a related transcript says: \"{transcript_segment}\"."
        if video_tags:
            context += f" This part of the transcript is associated with the keywords: '{', '.join(video_tags)}'. "
    elif SIMILARITY_THRESHOLD < similarity <= HIGH_SIMILARITY_THRESHOLD:
        context = f"There is a potential connection between your question and a segment in the video titled '{video_title}', from {start_time_formatted} to {end_time_formatted}. I couldn't find an exact match in the transcript or tags, though. "
        if video_tags:
            context += f"Despite this, the relevant segment is associated with the following keywords: '{', '.join(video_tags)}'. "
    else:
        context = ""

    full_prompt = context + "Given this, " + prompt

    response = openai.Completion.create(
        engine=model_engine,
        prompt=full_prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()


file_path = 'C:\\Users\\jpiye\\OneDrive\\Documents\\GitHub\\KATCHCapstone\\exec\\gpt3_5_executor\\formatted_response.json'

with open(file_path) as f:
    video_metadata = json.load(f)

if __name__ == "__main__":
    while True:
        user_input = input("Enter a prompt for KACH (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = interact_with_gpt3_5(user_input, video_metadata)
        print("KACH response:", response)
