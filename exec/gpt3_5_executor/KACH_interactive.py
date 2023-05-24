import json
import openai
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from yake import KeywordExtractor
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
    embeddings = outputs.last_hidden_state.detach().numpy()
    sentence_embedding = np.mean(embeddings, axis=1)
    return sentence_embedding[0]

def get_keywords(text):
    kw_extractor = KeywordExtractor(lan="en", n=3)
    keywords = kw_extractor.extract_keywords(text)
    return set(kw[0] for kw in keywords)

SIMILARITY_THRESHOLD = 0.4
def find_video_tags_and_transcript(prompt, video_metadata):
    words = get_keywords(prompt)
    prompt_embedding = get_embedding(prompt)
    best_match = ("", "", "", "", "", "", -1)

    for video in video_metadata["videos"]:
        for segment in video["transcript_portions"]:
            transcript_embedding = get_embedding(segment["text"])
            similarity = cosine_similarity([prompt_embedding], [transcript_embedding])[0][0]

            if similarity > best_match[6]:
                best_match = (video["title"], "", segment["text"], "", segment["start_time"], segment["end_time"], similarity)
                for tag in segment["tags"]:
                    if tag["word"].lower() in words:
                        best_match = (video["title"], tag["word"], segment["text"], "", segment["start_time"], segment["end_time"], similarity)

    return best_match

def interact_with_gpt3_5(prompt, video_metadata):
    model_engine = "text-davinci-003"
    video_title, video_tags, transcript_segment, _, _, _, similarity = find_video_tags_and_transcript(prompt, video_metadata)
    context = ""

    if similarity > SIMILARITY_THRESHOLD:
        context = f"In a video titled '{video_title}', a segment of the transcript reads: \"{transcript_segment}\". "
        if video_tags:
            context += f"This segment is tagged with '{video_tags}'. "

    full_prompt = context + prompt
    for _ in range(3): # Internal reprompting
        response = openai.Completion.create(
            engine=model_engine,
            prompt=full_prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_text = response.choices[0].text.strip()
        if response_text:
            break
        else:
            full_prompt += " Could you please provide more information?"

    return response_text

# Define the full path to your JSON file
file_path = 'C:\\Users\\jpiye\\OneDrive\\Documents\\GitHub\\KATCHCapstone\\exec\\gpt3_5_executor\\formatted_response.json'

# Now you can open the file with this path
with open(file_path) as f:
    video_metadata = json.load(f)

if __name__ == "__main__":
    while True:
        user_input = input("Enter a prompt for KACH (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = interact_with_gpt3_5(user_input, video_metadata)
        print("KACH response:", response)
