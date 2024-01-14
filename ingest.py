from haystack.nodes import PreProcessor, EmbeddingRetriever
from haystack.document_stores import PineconeDocumentStore
from haystack import Document
import os
from dotenv import load_dotenv
from haystack.utils import convert_files_to_docs
import pandas as pd
load_dotenv()


print("Import Successfully")

document_store = PineconeDocumentStore(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter",
                                       similarity="dot_product")

print("Document Store: ", document_store)
print("#####################")

docs = []

script_dir = os.path.dirname(__file__)
txt_folder_path = os.path.join('full_contract_txt')
# def extract_text_from_pdf(txt_path):
#    lines = ""
#    with open(txt_path, encoding='latin-1') as f:
#       lines = f.readlines()
   
#    return ' '.join(lines)

# for file in os.listdir(txt_folder_path):
#    file_path = os.path.join(txt_folder_path, file)
#    txt = extract_text_from_pdf(file_path)
#    doc = Document(content=str(txt),meta={"txt_path": file_path})

#    docs.append(doc)

docs = convert_files_to_docs(dir_path=txt_folder_path)

# Convert DataFrame content to string
for doc in docs:
    if isinstance(doc.content, pd.DataFrame):
        doc.content = doc.content.to_string(index=False)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=100,
    split_overlap=10,
    split_respect_sentence_boundary=True,
)
# print("Preprocessor: ", preprocessor)
print("#####################")

preprocessed_docs = preprocessor.process(docs)
# print("Preprocessed Docs: ", preprocessed_docs)

# Convert DataFrame content to string again after pre-processing
for doc in preprocessed_docs:
    if isinstance(doc.content, pd.DataFrame):
        doc.content = doc.content.to_string(index=False)

print("#####################")

document_store.write_documents(preprocessed_docs)

retriever = EmbeddingRetriever(document_store = document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

print("Retriever: ", retriever)

document_store.update_embeddings(retriever)

print("Embeddings Done.")

