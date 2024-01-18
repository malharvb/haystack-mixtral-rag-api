from haystack.nodes import EmbeddingRetriever, BM25Retriever, PreProcessor, AnswerParser, QuestionGenerator, PromptNode, PromptTemplate
from haystack.document_stores import PineconeDocumentStore, InMemoryDocumentStore
from haystack.utils import convert_files_to_docs, print_questions
from haystack import Pipeline
from haystack.pipelines import QuestionGenerationPipeline, RetrieverQuestionGenerationPipeline
from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from haystack.agents import Tool
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode
from haystack.agents import AgentStep, Agent
from haystack.agents.base import Agent, ToolsManager
import uvicorn
import json
import os
import glob
import time
from dotenv import load_dotenv


load_dotenv()


app = FastAPI()


# Configure templates
templates = Jinja2Templates(directory="templates")
HF_TOKEN = os.getenv('HF_TOKEN')


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
    # max_length=1000,
    model_kwargs={"model_max_length": 32000}
)
query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

search_tool = Tool(
    name="contract_clause_search",
    pipeline_or_node=query_pipeline,
    description="useful for when you need to answer questions about legal contract clauses",
    output_variable="answers",
)
agent_prompt_node = PromptNode(
    model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=HF_TOKEN,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5, "model_max_length": 32000},
)
memory_prompt_node = PromptNode(
    "philschmid/bart-large-cnn-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}
)
memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")
agent_prompt = """
In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
The AI Agent must use the available tools to find the up-to-date information. The final answer to the question should be truthfully based solely on the output of the tools. The AI Agent should ignore its knowledge when answering the questions.
The AI Agent has access to these tools:
{tool_names_with_descriptions}
The following is the previous conversation between a human and The AI Agent:
{memory}
AI Agent responses must start with one of the following:
Thought: [the AI Agent's reasoning process]
Tool: [tool names] (on a new line) Tool Input: [input as a question for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
Observation: [tool's result]
Final Answer: [final answer to the human user's question]
When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines.
The AI Agent should not ask the human user for additional information, clarification, or context.
If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive
Question: {query}
Thought:
{transcript}
"""

def resolver_function(query, agent, agent_step):
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }

conversational_agent = Agent(
    agent_prompt_node,
    prompt_template=agent_prompt,
    prompt_parameters_resolver=resolver_function,
    memory=memory,
    tools_manager=ToolsManager([search_tool]),
)


def get_result(documentType, legalClauses):

    query = f"Given a {documentType} which has the following clauses {legalClauses}. Return additional clauses that can be added to the legal agreement."

    json_response = conversational_agent.run(query=query, params={"Retriever" : {"top_k": 5}, "debug": True})

    for component in json_response['_debug']:
        print(component, json_response['_debug'][component]['exec_time_ms'])

    answers = json_response['answers']
    documents = json_response['documents']

    for ans in answers:
        answer = ans.answer
        break

    document_names = []

    for document in documents:
        document_names.append(document.id)


    return answer, document_names

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/doc-content")
# async def index(request: Request):
#     document_store.get_documents_by_id(ids=['49091c797d2236e73fab510b1e9c7f6b'], return_embedding=True)

@app.post("/rag")
async def get_answer(documentType: Annotated[str, Form()], legalClauses: Annotated[str, Form()]):
    start_time = time.time()
    answer, relevant_documents = get_result(documentType, legalClauses)
    response_data = jsonable_encoder(json.dumps({"answer": answer, "relevant_documents": relevant_documents}))
    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))
    res = Response(response_data)
    return res

@app.post("/doc-upload")
async def analyzeDoc(file: UploadFile = File(...)):
    start_time = time.time()
    filedir = 'docs'
    with open(f"{filedir}/{file.filename}", "wb") as f:
        content = await file.read()
        f.write(content)

    docs = convert_files_to_docs(dir_path=filedir)

    document_store = InMemoryDocumentStore(use_bm25=True)

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=100,
        split_overlap=10,
        split_respect_sentence_boundary=True,
    )

    preprocessed_docs = preprocessor.process(docs)


    document_store.write_documents(preprocessed_docs)

    retriever = BM25Retriever(document_store=document_store, top_k=10)

    prompt_template = PromptTemplate(
        prompt="""Synthesize a comprehensive answer from the following text for the given question.
                                Provide a clear and concise response that summarizes the key points and information presented in the text.
                                \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
        output_parser=AnswerParser(),
    )

    prompt_node = PromptNode(        
        model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_key=HF_TOKEN,
        default_prompt_template=prompt_template,
        max_length=1000,
        timeout= 300,
        model_kwargs={"model_max_length": 5000}
    )

    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

    documentType = pipe.run(query="Identify the type of legal contract in the given context. Answer only the type of legal contract and enclose it in curly braces. Add no other extra words to the answer.")

    print(documentType["answers"][0].answer)

    legalClauses = pipe.run(query="What legal clauses are present in the given legal document summarize them and state them below in a list.")

    print(legalClauses["answers"][0].answer)

    response_data = jsonable_encoder(json.dumps({"documentType": documentType["answers"][0].answer, "legalClauses": legalClauses["answers"][0].answer}))
    res = Response(response_data)


    files = glob.glob(f"{filedir}/*")
    for f in files:
        os.remove(f)

    print("Time took to process the request and return response is {} sec".format(time.time() - start_time))

    return res


@app.post("/question-gen")
async def analyzeDoc(file: UploadFile = File(...)):
    start_time = time.time()
    filedir = 'docs'
    with open(f"{filedir}/{file.filename}", "wb") as f:
        content = await file.read()
        f.write(content)

    docs = convert_files_to_docs(dir_path=filedir)

    document_store = InMemoryDocumentStore(use_bm25=True)

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="sentence",
        split_length=1,
        split_respect_sentence_boundary=False,
    )

    preprocessed_docs = preprocessor.process(docs)


    document_store.write_documents(preprocessed_docs)

    prompt_template = PromptTemplate(
        prompt="""
            Generate questions for the following statement, filling in the blanks represented by dots (.):
            Statement: {document}
        """,
        output_parser=AnswerParser(),
    )

    prompt_node = PromptNode(        
        model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_key=HF_TOKEN,
        default_prompt_template=prompt_template,
        max_length=1000,
        timeout= 300,
        model_kwargs={"model_max_length": 5000}
    )

    # question_generator = QuestionGenerator()

    # question_generation_pipeline = QuestionGenerationPipeline(question_generator)
    for idx, document in enumerate(document_store):

        print(f"\n * Generating questions for document {idx}:\n")
        # result = question_generation_pipeline.run(documents=[document])
        # print_questions(result)
        result = prompt_node.prompt(prompt_template=prompt_template, document=document)

        print(result)
        
    # retriever = BM25Retriever(document_store=document_store)
    # rqg_pipeline = RetrieverQuestionGenerationPipeline(retriever, question_generator)

    # print(f"\n * Generating questions for documents matching the query 'Arya Stark'\n")
    # result = rqg_pipeline.run(query=".....")
    # print_questions(result)

    files = glob.glob(f"{filedir}/*")
    for f in files:
        os.remove(f)

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8001, reload=True)



    # document_store = PineconeDocumentStore(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter",
    #                                 similarity="dot_product",
    #                                 embedding_dim=768)
    # retriever = EmbeddingRetriever(document_store = document_store,
    #                                 embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
    # prompt_template = PromptTemplate(prompt = """Answer the asked query based on only the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information.
    #                                             Documents: {join(documents)}
    #                                             Query: {query}
    #                                             Answer: 
    #                                         """,
    #                                 output_parser=AnswerParser()
    #                                 )
    # prompt_node = PromptNode(
    #     model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #     api_key=HF_TOKEN,
    #     default_prompt_template=prompt_template,
    #     max_length=1000,
    #     timeout= 300,
    #     model_kwargs={"model_max_length": 5000}
    # )
    # query_pipeline = Pipeline()
    # query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    # query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

    # query = f"Given a {documentType} which has the following clauses {legalClauses}. Return additional clauses that can be added to the legal agreement."

    # json_response = query_pipeline.run(query=query, params={"Retriever" : {"top_k": 5}, "debug": True})