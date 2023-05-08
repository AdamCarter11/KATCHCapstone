import json
from transformers import RagTokenizer, RagTokenForGeneration, RagConfig

def interact_with_rag(prompt, video_metadata, stored_transcript):
    model_name = "facebook/rag-token-nq"

    tokenizer = RagTokenizer.from_pretrained(model_name)
    config = RagConfig.from_pretrained(model_name)
    model = RagTokenForGeneration.from_pretrained(model_name, config=config)

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

    context = f"Transcript: \"{transcript}\"\n{prompt}"

    inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1,
                             temperature=5.0, top_k=10, top_p=0.95)
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
        user_input = input("Enter a prompt for RAG: ")
        response = interact_with_rag(user_input, video_metadata, stored_transcript)
        print("RAG Response:", response)
