from haystack.nodes import PreProcessor, EmbeddingRetriever
from haystack.document_stores import PineconeDocumentStore
from haystack import Document
import os
from dotenv import load_dotenv

load_dotenv()


print("Import Successfully")

document_store = PineconeDocumentStore(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter",
                                       similarity="dot_product",
                                       embedding_dim=768)

print("Document Store: ", document_store)
print("#####################")

docs = []

script_dir = os.path.dirname(__file__)
txt_folder_path = os.path.join('full_contract_txt')
def extract_text_from_pdf(txt_path):
   lines = ""
   with open(txt_path, encoding='latin-1') as f:
      lines = f.readlines()
   
   return ' '.join(lines)

for file in os.listdir(txt_folder_path):
   file_path = os.path.join(txt_folder_path, file)
   txt = extract_text_from_pdf(file_path)
   doc = Document(content=str(txt),meta={"txt_path": file_path})

   docs.append(doc)

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=768,
    split_respect_sentence_boundary=True,
)
# print("Preprocessor: ", preprocessor)
print("#####################")

preprocessed_docs = preprocessor.process(docs)
# print("Preprocessed Docs: ", preprocessed_docs)
print("#####################")

document_store.write_documents(preprocessed_docs)

retriever = EmbeddingRetriever(document_store = document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")

print("Retriever: ", retriever)

document_store.update_embeddings(retriever)

print("Embeddings Done.")

