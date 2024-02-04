from haystack import Pipeline
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate, PreProcessor
from haystack.utils import convert_files_to_docs, print_questions
import os
from dotenv import load_dotenv
from haystack.agents import Tool
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode
from haystack.agents import AgentStep, Agent
from haystack.agents.base import Agent, ToolsManager
from haystack.document_stores import PineconeDocumentStore, InMemoryDocumentStore
import time
import glob
import json

from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')

from datetime import datetime
 
# get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
print("Current date & time : ", current_datetime)
 
# convert datetime obj to string
str_current_datetime = str(current_datetime)
 
# create a file object along with extension
file_name = str_current_datetime+".txt"

########################################################################################################################
## No Context/History


# document_store = PineconeDocumentStore(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter",
#                                        similarity="dot_product",
#                                        embedding_dim=768)
              
# retriever = EmbeddingRetriever(document_store = document_store,
#                                embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
# prompt_template = PromptTemplate(prompt = """"Answer the question truthfully based solely on the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information. Your answer should be no longer than 1000 words. You should use no white space and only use bullet points.
#                                               not include an answer, reply with 'I don't know'. Extract legal clauses from the contexts and provide related answers. \n
#                                               Query: {query}\n
#                                               Documents: {join(documents)}
#                                               Answer: 
#                                           """)
# prompt_node = PromptNode(
#     model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     api_key=HF_TOKEN,
#     default_prompt_template=prompt_template,
#     max_length=500,
#     model_kwargs={"model_max_length": 5000}
# )
# query_pipeline = Pipeline()
# query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
# query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

# from pprint import pprint
# print_answer = lambda out: pprint(out["results"][0].strip())


# result = query_pipeline.run(query = "What legal clauses are there in a franchise contract", params={"Retriever" : {"top_k": 5}, "debug": True})

# for component in result['_debug']:
#     print(component, result['_debug'][component]['exec_time_ms'])

# result = query_pipeline.run(query = "What legal clauses are there in a leasing contract", params={"Retriever" : {"top_k": 5}, "debug": True})

# for component in result['_debug']:
#     print(component, result['_debug'][component]['exec_time_ms'])

########################################################################################################################
## Chat with documents trial with context/history

# start_time = time.time()

# HF_TOKEN = os.getenv('HF_TOKEN')

# document_store = PineconeDocumentStore(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter",
#                                     similarity="dot_product",
#                                     embedding_dim=768)
            
# retriever = EmbeddingRetriever(document_store = document_store,
#                             embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
# prompt_template = PromptTemplate(prompt = """Answer the question truthfully based solely on the given documents. If the documents do not contain the answer to the question, say that answering is not possible given the available information. Your answer should be no longer than 1000 words.
#                                             Documents: {join(documents)}
#                                             Query: {query}
#                                             Answer: 
#                                         """,
#                                 output_parser=AnswerParser())
# prompt_node = PromptNode(
#     model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     api_key=HF_TOKEN,
#     default_prompt_template=prompt_template,
#     # max_length=1000,
#     model_kwargs={"model_max_length": 32000}
# )
# query_pipeline = Pipeline()
# query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
# query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

# search_tool = Tool(
#     name="contract_clause_search",
#     pipeline_or_node=query_pipeline,
#     description="useful for when you need to answer questions about legal contract clauses",
#     output_variable="answers",
# )
# agent_prompt_node = PromptNode(
#     model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     api_key=HF_TOKEN,
#     stop_words=["Observation:"],
#     model_kwargs={"temperature": 0.5, "model_max_length": 32000},
# )
# memory_prompt_node = PromptNode(
#     "philschmid/bart-large-cnn-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}
# )
# memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")
# agent_prompt = """
# In the following conversation, a human user interacts with an AI Agent. The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers.
# The AI Agent must use the available tools to find the up-to-date information. The final answer to the question should be truthfully based solely on the output of the tools. The AI Agent should ignore its knowledge when answering the questions.
# The AI Agent has access to these tools:
# {tool_names_with_descriptions}
# The following is the previous conversation between a human and The AI Agent:
# {memory}
# AI Agent responses must start with one of the following:
# Thought: [the AI Agent's reasoning process]
# Tool: [tool names] (on a new line) Tool Input: [input as a question for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
# Observation: [tool's result]
# Final Answer: [final answer to the human user's question]
# When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines.
# The AI Agent should not ask the human user for additional information, clarification, or context.
# If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive
# Question: {query}
# Thought:
# {transcript}
# """

# def resolver_function(query, agent, agent_step):
#     return {
#         "query": query,
#         "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
#         "transcript": agent_step.transcript,
#         "memory": agent.memory.load(),
#     }

# conversational_agent = Agent(
#     agent_prompt_node,
#     prompt_template=agent_prompt,
#     prompt_parameters_resolver=resolver_function,
#     memory=memory,
#     tools_manager=ToolsManager([search_tool])
# )

# result = conversational_agent.run('Give clauses in a manufacturing contract')

# print("\nAssistant: ", result)

# print("Time took to process the request and return response is {} sec".format(time.time() - start_time))

# start_time = time.time()

# result = conversational_agent.run('Can you explain the first clause in detail and give an actual example of it')

# print("\nAssistant: ", result)

# print("Time took to process the request and return response is {} sec".format(time.time() - start_time))

import re

f= open(f"output_logs/{file_name}","w+")

def replace_dots_with_mask(input_string):

    # Replace continuous sequence of dots with [MASK]
    # result_string = re.sub(r'\.{3,}', ' [MASK] ', input_string)

    counter = 1
    result_string = ''
    start = 0
    for m in re.finditer(r'\.{3,}|_{3,}', input_string):
        end, newstart = m.span()
        result_string += input_string[start:end]
        rep = f" [MASK{counter}] "
        result_string += rep
        start = newstart
        counter += 1
    result_string += input_string[start:]

    return result_string.strip()

filedir = 'docs'

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
# prompt_template = PromptTemplate(
#     prompt="""
#     Draft questions that help the user understand the [MASK] tokens present in the given sentence. A question should be easy to understand and must be based on only one [MASK] token. Return the only one question for one [MASK] token as a list in the answer.
#     Sentence: {document}
#     """,
#     output_parser=AnswerParser(),
# )

# prompt_template = PromptTemplate(
#     prompt="""
#     Identify each [MASK] token present in the sentence. Explain to the user what each [MASK] token represents in less than 15 words. The answer should only be a list of these explanations for example [MASK] - Explanation \n 
#     Sentence: {document.content}\n
#     Answer:
#     """,
#     output_parser=AnswerParser(),
# )
# Generate fill-in-the-blank questions for the following sentence. Replace [MASK] with appropriate terms. Return the questions as a list.

# prompt_template = PromptTemplate(
#     prompt="""
#     Replace only [MASK] token with appropriate type of data that should be present in square brackets. Only do what you are asked. Return the modified sentence\n 
#     Sentence: {query}\n
#     Answer:
#     """,
#     output_parser=AnswerParser(),
# )



# prompt_template = PromptTemplate(
#     prompt="""
#     You are a legal contract based question drafting assistant. Draft questions for the fill in the blanks in the given sentence. The blanks are represented by [MASK] token. The length of a question should be less than 15 words. Drafted questions must not contain [MASK] token in them. The questions should be returned along with [MASK] token number.
#     Sentence: {document}\n
#     """,
#     output_parser=AnswerParser(),
# )

## Current best
# prompt_template = PromptTemplate(
#     prompt="""
#     Explain what must the [MASK] tokens in the sentence be filled with in easy to understand words in less than 15 words. The explantion must contain intent of [MASK] token only. Make sure that there is only a single explanation for an unique [MASK] token. Return the answer as a json object where the keys are the [MASK] tokens and the values are the explanations. 
#     Sentence: {document}
#     """,
#     output_parser=AnswerParser(),
# )

prompt_template = PromptTemplate(
    prompt="""
    Generate a single, clear question for each [MASK1], [MASK2], [MASK3], etc., in the document. Ensure that each question is uniquely tailored to gather information for the specific [MASK] it corresponds to. Do not generate any extra questions unrelated to the specified placeholders. Return the answer as a json object where the keys are the [MASK] tokens and the values are the questions. 
    Document: {document}
    """,
    output_parser=AnswerParser(),
)


prompt_node = PromptNode(        
    model_name_or_path="gpt-3.5-turbo",
    api_key=OPEN_AI_KEY,
    # model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    # api_key=HF_TOKEN,
    default_prompt_template=prompt_template,
    # output_variable="my_answer",
    model_kwargs={"temperature": 0.2, "top_p": 0.1, "max_tokens": 2000}
)

# question_prompt_template = PromptTemplate(
#     prompt="""
#     Generate only a single direct question for every token in square brackets. Return the list of questions\n 
#     Sentence: {my_answer}\n
#     Answer:
#     """,
#     output_parser=AnswerParser(),
# )

# question_prompt_node = PromptNode(        
#     model_name_or_path="gpt-3.5-turbo-instruct",
#     api_key=OPEN_AI_KEY,
#     # model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
#     # api_key=HF_TOKEN,
#     default_prompt_template=question_prompt_template,
#     model_kwargs={"temperature": 0.1, "max_new_tokens": 2048, "top_p": 0.9, "repetition_penalty": 1.2}
# )

# pipe = Pipeline()
# pipe.add_node(component=prompt_node, name="prompt_node", inputs=["Query"])
# pipe.add_node(component=question_prompt_node, name="question_prompt_node", inputs=["prompt_node"])

for idx, document in enumerate(document_store):
    if idx != 3:
        continue
    print(f"\n * Generating questions for document {idx}:\n")

    blankspaces = re.findall(r'\.{3,}|_{3,}', document.content)
    print(blankspaces)

    if not blankspaces:
        continue


    document.content = replace_dots_with_mask(document.content)

    result = prompt_node.prompt(prompt_template=prompt_template, document=document.content)
    print(result[0].answer)
    print(result[0].meta)

    answerJson = json.loads(result[0].answer)
    removeItems = []
    for key, value in answerJson.items():
        if key not in document.content:
            removeItems.append(key)
    
    for key in removeItems:
        answerJson.pop(key)
    f.write(str(json.dumps(answerJson)) + '\n')
