import os
from dotenv import load_dotenv
from typing import List, Optional
from llama_index.llms.groq import Groq

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, FilterCondition


PERSIST_DIR = "./storage"

def get_doc_tools(
    file_path: str,
    name: str,
):
    """
    Get vector query and summary query tools from a document with persistence.
    :param file_path: path of file/doc
    :param name: name of tool
    :return: returns Vector Query tool and  Summary tool
    """
    vector_index = None

    if not os.path.exists(PERSIST_DIR):
        print("Creating data base !!")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes)

        vector_index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:

        print("Using stored data !!")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        vector_index = load_index_from_storage(storage_context)

    if vector_index is None:
        raise ValueError("VectorStoreIndex is not properly initialized.")

    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """
        Get vector query tool from a document with persistence used for indexed retrieval.
        :param query: input query for generation
        :param page_numbers: optional page numbers to reduce search span of retrieval
        :return: generate response
        """
        page_numbers = page_numbers or []
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
        
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    # To create a summary_index, we need nodes, which are part of the documents, not vector_index
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # summary_index = SummaryIndex(nodes)
    # summary_query_engine = summary_index.as_query_engine(
    #     response_mode="tree_summarize",
    #     use_async=True,
    # )
    # summary_tool = QueryEngineTool.from_defaults(
    #     name=f"summary_tool_{name}",
    #     query_engine=summary_query_engine,
    #     description=f"Useful for summarization questions related to {name}",
    # )

    return vector_query_tool

def clean_text(text: str) -> str:
    """
    Clean the raw text by removing unnecessary whitespace and new lines.
    :param text: input unclean text 
    :return: cleaned text
    """
    cleaned_text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
    return cleaned_text

load_dotenv()

def initialize_settings():
    """
    Used to initialize LLM, Embedding Model and Sentence splitter.
    """
    try : 
        Settings.llm = Groq(model="llama3-8b-8192")
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    except : 
        Settings.llm = Groq(model="llama3-8b-8192", api_key=input("enter api key"))
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)


