import spacy
import json
#spacy.cli.download("en_core_web_sm")


class SentenceConverter:
    def __init__(self):
        """
        Initializes the SentenceConverter class.
        Loads the Spacy language model for sentence tokenization.
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.converted_videos = {"videos": []}

    def run(self, videos):
        """
        Processes the input data and returns the converted videos with added transcript portions.

        Args:
            videos (dict or list): The input data, which can be a dictionary or a list of dictionaries.

        Returns:
            list: The converted videos with added transcript portions.
        """
        if isinstance(videos, dict):
            self.run_video(videos['data'])
        elif isinstance(videos, list):
            for video in videos:
                self.run_video(video['data'])
        else:
            raise ValueError("Invalid data type. Expected a list or dictionary.")

        return self.converted_videos

    def run_video(self, video):
        """
        Processes an individual video and adds the converted data to the converted_videos list.

        Args:
            video (dict): The video data to be processed.
        """
        converted_data = {
            "title": video["title"],
            "publish_date": video["published"],
            "duration": video["duration"],
            "youtube_id": video["youtube_id"],
            "channel_title": video["channel"]["title"],
            "thumbnail_url": video["thumbnail_url"],
            "transcript_portions": self.convert_and_add_tags(video)
        }
        self.converted_videos["videos"].append(converted_data)

    def convert_and_add_tags(self, video):
        """
        Converts the video's word list into sentences and adds tags to the sentences.

        Args:
            video (dict): The video data containing word list and tag list.

        Returns:
            list: The list of sentences with added tags.
        """
        word_list = video['full_tagstream']['srts']
        tag_list = video['full_tagstream']['tags']

        sentences = []
        current_sentence = []
        unique_speakers = set()

        for word_dict in word_list:
            current_sentence.append(word_dict)
            if 'ending_punctuation' in word_dict:
                word_dict['word'] = word_dict['word'] + word_dict['ending_punctuation']
                if word_dict['ending_punctuation'] in ['.', '!', '?']:
                    sentence_text = ' '.join([word['word'] for word in current_sentence])
                    sentence_tokens = self.nlp(sentence_text)
                    for sent in sentence_tokens.sents:
                        for word in current_sentence:
                            speaker_tuple = tuple(word['speaker'].items())
                            unique_speakers.add(speaker_tuple)
                        sentence_dict = {
                            'text': sent.text,
                            'start_time': current_sentence[0]['timestamp'],
                            'end_time': sum([word['length'] for word in current_sentence]) + float(
                                current_sentence[0]['timestamp']),
                            'speakers': [dict(speaker_tuple) for speaker_tuple in unique_speakers],
                        }
                        sentences.append(sentence_dict)
                    current_sentence = []
                    unique_speakers.clear()

        for sentence in sentences:
            matching_tags = []
            for tag in tag_list:
                for position in tag['position']:
                    if float(sentence['start_time']) <= float(position['timestamp']) <= float(sentence['end_time']):
                        matching_tags.append(tag)
                        break

            updated_tags = []
            for tag in matching_tags:
                positions = [position for position in tag['position'] if
                             float(sentence['start_time']) <= float(position['timestamp']) <= float(
                                 sentence['end_time'])]
                for position in positions:
                    updated_tag = {
                        'word': tag['tag'],
                        'wordIndex': position['wordIndex'],
                        'timestamp': position['timestamp'],
                    }
                    if updated_tag not in updated_tags:
                        updated_tags.append(updated_tag)

            sentence['tags'] = sorted(updated_tags, key=lambda x: (x.get('wordIndex', float('inf')), x['timestamp']))

        return sentences


# Read the JSON file
with open('response.json', 'r') as file:
    file_contents = file.read()

# Parse the JSON content
data = json.loads(file_contents)
file.close()
# create a class instance and run the converter
# NOTE: The run method can take a list of videos or a single video object
converter = SentenceConverter()
combined_results = converter.run(data)

# Print the resulting sentences
#print(json.dumps(combined_results, indent=4))
# Open the file in write mode
file = open('formatted_response.json', 'w')

# Write content to the file
file.write(json.dumps(combined_results, indent=4))


# Close the file
file.close()

'''
The above uses a file on your system if it is in the same filepath.

This method can handle responses directly from the DCO API as well.

To do this, you can use the following code:


import DcoApi
import SentenceConverter

dcoapi = DcoApi()
converter = SentenceConverter()

#  This gets a list of all your TagStreams
resp = dcoapi.send('GET', '/api/jpiyer@ucsc.edu/tag_streams')

#  Then we loop and get the individual TagStreams
TagStreams = []
for tag_stream in resp['data']:
    TagStream = dcoapi.send('GET', '/api/users/tag_streams/tag_stream['id'])
    TagStreams.append(TagStream)

#  Now run them through the converter
combined_results = converter.run(TagStreams)

# Print the resulting sentences
print(json.dumps(combined_results, indent=4))
'''