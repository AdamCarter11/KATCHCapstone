from jina import Executor, requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPTExecutor(Executor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    @requests
    def generate_text(self, docs, *args, **kwargs):
        for doc in docs:
            transcript = doc.tags['transcript']
            video_tags = doc.tags['video_tags']
            timestamps = doc.tags['timestamps']

            input_text = f"Transcript: {transcript}. Video Tags: {', '.join(video_tags)}. Timestamps: {', '.join(map(str, timestamps))}."

            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            doc.tags['generated_text'] = generated_text

