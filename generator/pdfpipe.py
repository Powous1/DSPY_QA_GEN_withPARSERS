from pathlib import Path
from pdfparse import parse_list,isjunk
from chonkie import TokenChunker, SentenceChunker, \
    RecursiveChunker, SemanticChunker, NeuralChunker 

CHUNKER_BUILDERS = {
    "recursive": 
    lambda size: RecursiveChunker(chunk_size=size),
    "token": 
    lambda size: TokenChunker(chunk_size=size),
    "semantic": 
    lambda size: SemanticChunker(),
    "sentence":
    lambda size: SentenceChunker(),
    "neural": 
    lambda size: NeuralChunker(),
}

_CHUNKER_CACHE = {}

def get_chunker(name: str, chunk_size: int):
    key = (name, chunk_size)
    if key not in _CHUNKER_CACHE:
        builder = CHUNKER_BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"Unknown chunk strategy '{name}'. Valid: {list(CHUNKER_BUILDERS.keys())}")
        _CHUNKER_CACHE[key] = builder(chunk_size)
    return _CHUNKER_CACHE[key]


class create_context:

    def __init__(self, parser, pdf_file, 
                 rows, chunk, chunk_size):
        self.file = pdf_file
        self.size = int(rows)
        self.p = parse_list.get(parser)
        if self.p is None:
            raise ValueError(f"Unknown parser '{parser}'. Valid: {list(parse_list.keys())}")
        self.chunk = get_chunker(chunk, chunk_size)   # only builds what you actually asked for

    def get_tables(self):
        pdf = self.p['open'](self.file)
        try:
            meta = self.p['meta'](pdf)
            data = (meta.get("subject") or "").split()
            title = (meta.get("title") or f"{self.file}").replace("-", "_")
            date = data[-1].replace("-", "_") if data else "unknown"
            type = data[1].replace("-", "_") if len(data) > 1 else "unknown"
            tables = [
                {"doc_id": f"{title}_created_in_({date})", "page_num": f"{i + 1}",
                    "table": chunked.text, "type": f"{type}"}
                for i, page in enumerate(self.p['pages'](pdf))
                for text in [self.p['content'](page)]
                if text is not None
                for chunked in self.chunk(text)
                if not isjunk(chunked.text)
            ][:self.size]
        finally:
            close_fn = self.p.get('close')
            if close_fn:
                close_fn(pdf)
        return tables

