import sys  # used to see if there are command line args

import docarray  # , weaviate
import jina
from jina import Flow


#  Print information about the Env Versions
print('docarray.__version__=', docarray.__version__)
print('jina.__version__=', jina.__version__)

f = Flow.load_config('flow.yml')

# for passing test using the Command Line.
if len(sys.argv) > 1:
    if sys.argv[1] == "svg":
        f.plot('flow.svg')  # this command will create a flow diagram in the root directory

with f:
    f.post(on='/')  # This starts the execution
    f.block()  # keeps the script running for the API calls

# Nothing can happen below here since block() halts any further execution until after the scripts stop.
print('\nGoodbye, Your Jina API is no longer running.\n')
#hello