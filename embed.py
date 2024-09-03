import os
import json
import uuid

from langchain.document_loaders import (
    BSHTMLLoader,
    DirectoryLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
load_dotenv()

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "business_data"

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    if os.path.exists("./chroma"):
        print("already embedded")
        exit(0)

    loader = DirectoryLoader(
        "./scrape",
        glob="*.html",
        loader_cls=BSHTMLLoader,
        show_progress=True,
        loader_kwargs={"get_text_separator": " "},
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    data = loader.load()
    documents = text_splitter.split_documents(data)

    # map sources from file directory to web source
    with open("./scrape/sitemap.json", "r") as f:
        sitemap = json.loads(f.read())

    business_id = uuid.uuid4()

    for document in documents:
        document.metadata["source"] = sitemap[
            document.metadata["source"].replace(".html", "").replace("scrape/", "")
        ]
        document.metadata["business_id"] = business_id

    # embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    # db = Chroma.from_documents(documents, embedding_model, persist_directory="./chroma")
    # db.persist()
    ids = vector_store.add_documents(documents)
    print(ids)
    print("\n\nbusiness_id", business_id)
