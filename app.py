from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, AnswerParser, PromptModel, PromptNode, PromptTemplate
from haystack.document_stores import PineconeDocumentStore
from haystack import Pipeline

from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import uvicorn
import json
import re
import os
from dotenv import load_dotenv
from haystack.agents import Tool
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode
from haystack.agents import AgentStep, Agent
from haystack.agents.base import Agent, ToolsManager


load_dotenv()

app = FastAPI()

HF_TOKEN = os.getenv('HF_TOKEN')

# Configure templates
templates = Jinja2Templates(directory="templates")

def get_result(query):

    document_store = PineconeDocumentStore(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter",
                                        similarity="dot_product",
                                        embedding_dim=768)
                
    retriever = EmbeddingRetriever(document_store = document_store,
                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    prompt_template = PromptTemplate(prompt = """Answer the question truthfully based solely on the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information. Your answer should be no longer than 1000 words.
                                                Documents: {join(documents)}
                                                Query: {query}
                                                Answer: 
                                            """,
                                    output_parser=AnswerParser())
    prompt_node = PromptNode(
        model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_key=HF_TOKEN,
        default_prompt_template=prompt_template,
        max_length=1000,
        model_kwargs={"model_max_length": 5000}
    )
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
    json_response = query_pipeline.run(query = str(query).strip(), params={"Retriever" : {"top_k": 2}})

    print("Answer: ", json_response)
    answers = json_response['answers']
    for ans in answers:
        answer = ans.answer
        break

    # Extract relevant documents and their content
    documents = json_response['documents']
    document_info = []

    for document in documents:
        content = document.content
        document_info.append(content)

    # Print the extracted information
    print("Answer:")
    print(answer)
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<=[.!?])\s', answer)

    # Filter out incomplete sentences
    # complete_sentences = [sentence for sentence in sentences if re.search(r'[.!?]$', sentence)]

    # Rejoin the complete sentences into a single string
    updated_answer = ' '.join(sentences)

    relevant_documents = ""
    for i, doc_content in enumerate(document_info):
        relevant_documents+= f"Document {i + 1} Content:"
        relevant_documents+=doc_content
        relevant_documents+="\n"

    print("Relevant Documents:", relevant_documents)

    return updated_answer, relevant_documents

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/rag")
async def get_answer(request: Request, question: str = Form(...)):
    print(question)
    answer, relevant_documents = get_result(question)
    response_data = jsonable_encoder(json.dumps({"answer": answer, "relevant_documents": relevant_documents}))
    res = Response(response_data)
    return res

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8001, reload=True)