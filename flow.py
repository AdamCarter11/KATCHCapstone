import sys  # used to see if there are command line args
import docarray  # , weaviate
import jina
from jina import Flow, Document, DocumentArray

def print_responses(resp):
    for doc in resp.docs:
        print(f"Generated Text: {doc.tags['generated_text']}")

doc1 = Document()
doc1.tags['transcript'] = 'Sample transcript text.'
doc1.tags['video_tags'] = ['tag1', 'tag2', 'tag3']
doc1.tags['timestamps'] = [1, 2, 3]

docs = DocumentArray([doc1])

#  Print information about the Env Versions
print('docarray.__version__=', docarray.__version__)
print('jina.__version__=', jina.__version__)

f = Flow.load_config('flow.yml')

# for passing test using the Command Line.
if len(sys.argv) > 1:
    if sys.argv[1] == "svg":
        f.plot('flow.svg')  # this command will create a flow diagram in the root directory

with f:
    f.post(on='/generate', inputs=docs, on_done=print_responses)
    f.block()  # keeps the script running for the API calls

# Nothing can happen below here since block() halts any further execution until after the scripts stop.
print('\nGoodbye, Your Jina API is no longer running.\n')
