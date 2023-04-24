from jina import Executor, requests, Document, DocumentArray
class ImageObjectIdentification(Executor):
    '''
    This is an example Executor. Here you would add all the functionality needed
    for this to work.
    '''

    def __init__(
            self,
            traversal_paths: str = '@r,c,m',
            **kwargs
    ):
        pass
    