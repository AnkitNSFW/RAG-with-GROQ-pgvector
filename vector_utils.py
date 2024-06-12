from sentence_transformers import SentenceTransformer
import numpy as np
from utils import scrape_wiki_content

# Our 384-Dimension Encoder
model = SentenceTransformer('thenlper/gte-small')

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text_data)
    return chunks

if __name__=='__main__':
    text = scrape_wiki_content('https://en.wikipedia.org/wiki/Spider-Man')
    chunks = split_text(text_data=text)
    embedding = model.encode(chunks)
