"""Indexing: PDF -> pages -> chunks -> embeddings -> FAISS vector index."""
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.get_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdfs(files: list[tuple[str, bytes]]) -> list[Document]:
    """Read uploaded PDFs, given as (file name, bytes), into one Document per page."""
    pages = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, data in files:
            path = Path(tmp_dir) / Path(name).name
            path.write_bytes(data)
            for page in PyPDFLoader(str(path)).load():
                page.metadata["source"] = name  # the original file name, not the temp path
                pages.append(page)
    return pages


def split_pages(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return [chunk for chunk in splitter.split_documents(pages) if chunk.page_content.strip()]


def build_index(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    # The embeddings are normalized, so inner product = cosine similarity
    return FAISS.from_documents(
        chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
