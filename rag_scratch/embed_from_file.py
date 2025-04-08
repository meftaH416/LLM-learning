import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Auto-install missing libraries
REQUIRED_LIBS = {
    "PyMuPDF": "1.23.26",
    "pymupdf4llm": "",
    "matplotlib": "3.8.3",
    "numpy": "1.26.4",
    "pandas": "2.2.1",
    "requests": "2.31.0",
    "sentence_transformers": "2.5.1",
    "spacy": "",
    "tqdm": "4.66.2",
    "transformers": "4.38.2",
    "accelerate": "",
    "bitsandbytes": "",
}

for lib, version in REQUIRED_LIBS.items():
    try:
        __import__(lib if lib != "sentence_transformers" else "sentence_transformers")
    except ImportError:
        os.system(f"pip install {lib}=={version}" if version else f"pip install {lib}")

import pymupdf4llm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer

class EmbeddingFromFile:
    def __init__(self, pdf_path, embed_path, device="cuda"):
        self.pdf_path = pdf_path
        self.embed_path = embed_path
        self.device = device

    def markeddown_text(self, page_chunks=True, show_progress=True):
        self.doc_md_texts = []  # list of tuples: (doc_name, md_list)
        if isinstance(self.pdf_path, list):
            for pdf in self.pdf_path:
                doc_name = os.path.basename(pdf)
                md = pymupdf4llm.to_markdown(
                    doc=pdf, page_chunks=page_chunks, show_progress=show_progress)
                for item in md:
                    item["document_name"] = doc_name
                self.doc_md_texts.append((doc_name, md))
        else:
            doc_name = os.path.basename(self.pdf_path)
            md = pymupdf4llm.to_markdown(
                doc=self.pdf_path, page_chunks=page_chunks, show_progress=show_progress)
            for item in md:
                item["document_name"] = doc_name
            self.doc_md_texts.append((doc_name, md))
        return self.doc_md_texts

    def page_and_text(self, doc_md_texts, first_page=0):
        self.pages_and_texts = []
        nlp = English()
        nlp.add_pipe("sentencizer")

        for doc_name, md_text in doc_md_texts:
            for i, val in enumerate(tqdm(md_text[first_page:], desc=f"Processing {doc_name}")):
                page_num = i + 1  # Start from 1 for each document
                content = val['text'].replace("\n", " ").replace("#", "").replace("*", "").strip()
                sentences = [str(sent) for sent in list(nlp(content).sents)]

                self.pages_and_texts.append({
                    "document_name": doc_name,
                    "page_num": page_num,
                    "page_character_count": len(content),
                    "page_word_count": len(content.split()),
                    "page_sentence_count": len(sentences),
                    "page_token_count": int(len(content) / 4),
                    "sentence": sentences,
                    "content": content
                })
        return self.pages_and_texts

    def create_chunk(self, pages_and_texts, slice_size=8):
        self.pages_and_chunks = []
        for item in tqdm(pages_and_texts, desc="Creating Chunks"):
            sentence_chunks = [item["sentence"][i:i + slice_size]
                               for i in range(0, len(item["sentence"]), slice_size)]
            item["sentence_chunks"] = sentence_chunks
            item["num_chunks"] = len(sentence_chunks)

            for sentence_chunk in sentence_chunks:
                chunk_dict = {
                    "document_name": item["document_name"],
                    "page_number": item["page_num"],
                    "sentence_chunk": re.sub(r'\.([A-Z])', r'. \1', " ".join(sentence_chunk).strip()),
                }
                chunk_dict["chunk_char_count"] = len(chunk_dict["sentence_chunk"])
                chunk_dict["chunk_word_count"] = len(chunk_dict["sentence_chunk"].split())
                chunk_dict["chunk_token_count"] = len(chunk_dict["sentence_chunk"]) / 4

                self.pages_and_chunks.append(chunk_dict)
        return self.pages_and_chunks

    def df_with_min_chunk_length(self, pages_and_chunks, min_token_length=5):
        self.df = pd.DataFrame(pages_and_chunks)
        self.pages_and_chunks_over_min_token_len = self.df[
            self.df["chunk_token_count"] > min_token_length
        ].to_dict(orient="records")
        return self.pages_and_chunks_over_min_token_len

    def sentence_embeddings(self, pages_and_chunks_over_min_token_len, model="all-mpnet-base-v2"):
        self.embedding_model = SentenceTransformer(model_name_or_path=model, device=self.device)

        for item in tqdm(pages_and_chunks_over_min_token_len, desc="Generating Embeddings"):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])

        self.text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
        self.text_chunks_and_embeddings_df.to_csv(self.embed_path, index=False)
        return self.text_chunks_and_embeddings_df

    def plot_stats(self, embeddings_df_save_path=None):
        if embeddings_df_save_path:
            df_plot = pd.read_csv(embeddings_df_save_path)
        elif hasattr(self, 'text_chunks_and_embeddings_df'):
            df_plot = self.text_chunks_and_embeddings_df
        else:
            raise ValueError("No data to plot. Provide a path or run `sentence_embeddings()` first.")

        plt.figure(figsize=(10, 6))
        plt.plot(df_plot.index, df_plot['chunk_token_count'], color='k', label="Token Count per Chunk")
        plt.axhline(df_plot['chunk_token_count'].mean(), color='red', linestyle='--', label='Mean Token Count')
        plt.xlabel("Chunk Index")
        plt.ylabel("Number of Tokens per Chunk")
        plt.legend()
        plt.grid(False)
        plt.show()


## use the Class
## Call the class
# embedder = EmbeddingFromFile(pdf_path="path/to/file.pdf", embed_path="output/embeddings.csv")
## Create markeddown text using pymupdf4llm
# md_text = embedder.markeddown_text()
## Creating dictionary with page num, token num, content and sentences using NLP LLM model
# pages_text = embedder.page_and_text(md_text,first_page=24)
## Creating chunked
# chunked = embedder.create_chunk(pages_text, slice_size=10)
## Filtered dataset excluding chunks less than certain number of tokens
# filtered = embedder.df_with_min_chunk_length(chunked, min_token_length=5)
## Creating embedding and pandas dataframe
# df_with_embeddings = embedder.sentence_embeddings(filtered,model="all-mpnet-base-v2")
## Plotting the stats i.e., chunk index vs token per chunks
# embedder.plot_stats()

