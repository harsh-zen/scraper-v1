# import argparse
# import os
# import json
# import uuid
# from langchain_community.document_loaders import AsyncChromiumLoader
# from langchain_community.document_transformers import BeautifulSoupTransformer
# from langchain_openai import OpenAIEmbeddings
# from langchain_postgres.vectorstores import PGVector
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# from langchain_community.document_loaders import AsyncHtmlLoader
# # Load environment variables
# load_dotenv()

# # Database connection parameters
# connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
# collection_name = "business_data"

# # Setup embeddings and vector store
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# vector_store = PGVector(
#     embeddings=embeddings,
#     collection_name=collection_name,
#     connection=connection,
#     use_jsonb=True,
# )

# # Setup command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--site", type=str, required=True)
# parser.add_argument("--depth", type=int, default=3)

# def scrape_and_transform(url: str):
#     # Initialize AsyncChromiumLoader
#     loader = AsyncHtmlLoader([url])
#     html = loader.load()

#     # Initialize BeautifulSoupTransformer
#     bs_transformer = BeautifulSoupTransformer()
#     docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p", "li", "div", "a", "span"])
    
#     return docs_transformed

# def main():
#     args = parser.parse_args()
#     url = args.site
    
#     # Scrape and transform documents
#     docs_transformed = scrape_and_transform(url)
    
#     # Add documents to PGVector
#     business_id = str(uuid.uuid4())
#     for document in docs_transformed:
#         document.metadata["business_id"] = business_id
    
#     ids = vector_store.add_documents(docs_transformed)
#     print("Document IDs:", ids)
#     print("Business ID:", business_id)

# if __name__ == "__main__":
#     main()
import argparse
import os
import json
import uuid
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Database connection parameters
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "business_data"

# Setup embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# Setup command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--site", type=str, required=True)
parser.add_argument("--depth", type=int, default=3)

def is_valid_url(url, base_url):
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return parsed_url.netloc == parsed_base.netloc and parsed_url.scheme in ['http', 'https']

async def scrape_and_transform(url: str, depth: int, base_url: str):
    if depth <= 0:
        return []

    print(f"Scraping URL: {url} (Depth: {depth})")
    
    # Initialize AsyncHtmlLoader
    loader = AsyncHtmlLoader([url])
    html = await loader.aload()
    
    # Initialize BeautifulSoupTransformer
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p", "li", "div", "a", "span"])
    
    all_docs = docs_transformed

    if depth > 1:
        # Create BeautifulSoup object directly
        soup = BeautifulSoup(html[0].page_content, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]
        unique_links = list(set(links))
        
        for link in unique_links:
            full_url = urljoin(base_url, link)
            if is_valid_url(full_url, base_url):
                nested_docs = await scrape_and_transform(full_url, depth - 1, base_url)
                all_docs.extend(nested_docs)
    
    return all_docs

async def main():
    args = parser.parse_args()
    url = args.site
    depth = args.depth
    
    # Scrape and transform documents
    docs_transformed = await scrape_and_transform(url, depth, url)
    
    # Add documents to PGVector
    business_id = str(uuid.uuid4())
    for document in docs_transformed:
        document.metadata["business_id"] = business_id
    
    ids = vector_store.add_documents(docs_transformed)
    print(f"Added {len(ids)} documents to the vector store")
    print("Business ID:", business_id)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())