import os
import tempfile
import uuid
import pandas as pd
import re

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import ChatOllama
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


def clean_filename(filename):
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename


def get_pdf_text(uploaded_file):
    """Load PDF into documents"""
    try:
        input_file = uploaded_file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents
    finally:
        os.unlink(temp_file.name)


def split_document(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    """Use Ollama embeddings (defaults to `nomic-embed-text`)"""
    return OllamaEmbeddings(model="nomic-embed-text")  # you can change model


def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    unique_ids = set()
    unique_chunks = []
    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        collection_name=clean_filename(file_name),
        embedding=embedding_function,
        ids=list(unique_ids),
        persist_directory=vector_store_path
    )
    vectorstore.persist()
    return vectorstore


def create_vectorstore_from_texts(documents, file_name):
    docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    embedding_function = get_embedding_function()
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    return vectorstore


def load_vectorstore(file_name, vectorstore_path="db"):
    embedding_function = get_embedding_function()
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function,
        collection_name=clean_filename(file_name)
    )


# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""


class AnswerWithSources(BaseModel):
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")


class ExtractedInfoWithSources(BaseModel):
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query_document(vectorstore, query):
    """Query the vectorstore and return structured response"""
    llm = ChatOllama(model="mistral")  # change to llama2, qwen, etc.

    retriever = vectorstore.as_retriever(search_type="similarity")
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    )

    structured_response = rag_chain.invoke(query)
    df = pd.DataFrame([structured_response.dict()])

    answer_row, source_row, reasoning_row = [], [], []
    for col in df.columns:
        answer_row.append(df[col][0]['answer'])
        source_row.append(df[col][0]['sources'])
        reasoning_row.append(df[col][0]['reasoning'])

    structured_response_df = pd.DataFrame(
        [answer_row, source_row, reasoning_row],
        columns=df.columns,
        index=['answer', 'source', 'reasoning']
    )
    return structured_response_df.T
