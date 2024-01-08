from haystack import Pipeline
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate
from haystack.utils import print_answers
import os
from dotenv import load_dotenv
from haystack.agents import Tool
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode
from haystack.agents import AgentStep, Agent
from haystack.agents.base import Agent, ToolsManager
from haystack.document_stores import PineconeDocumentStore

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

document_store = PineconeDocumentStore(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter",
                                       similarity="dot_product",
                                       embedding_dim=768)
              
retriever = EmbeddingRetriever(document_store = document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
prompt_template = PromptTemplate(prompt = """"Answer the following query based on the provided context. Their are multiple contracts in the context. If the context does
                                              not include an answer, reply with 'I don't know'. Extract legal clauses from the contexts and provide related answers. \n
                                              Query: {query}\n
                                              Documents: {join(documents)}
                                              Answer: 
                                          """)
prompt_node = PromptNode(
    model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=HF_TOKEN,
    default_prompt_template=prompt_template,
    max_length=500,
    model_kwargs={"model_max_length": 5000}
)
query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

from pprint import pprint
print_answer = lambda out: pprint(out["results"][0].strip())


print_answer(query_pipeline.run(query = "What is a legal contract", params={"Retriever" : {"top_k": 2}}))
print_answer(query_pipeline.run(query = "What legal clauses are their in a leasing contract", params={"Retriever" : {"top_k": 2}}))



#####################################################################################################################
## Chat with documents trial

# HF_TOKEN = os.getenv('HF_TOKEN')
# print(HF_TOKEN)


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
#     model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
#     api_key=HF_TOKEN,
#     default_prompt_template=prompt_template,
#     # max_length=1000,
#     # model_kwargs={"model_max_length": 5000}
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
#     model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1",
#     api_key=HF_TOKEN,
#     stop_words=["Observation:"],
#     model_kwargs={"temperature": 0.5},
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
#     tools_manager=ToolsManager([search_tool]),
# )

# print(conversational_agent.run('What are the clauses in a legal contract'))