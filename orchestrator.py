from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import ArxivAPIWrapper
from langchain_tavily import TavilySearch
from langgraph.func import entrypoint, task
from langchain.chat_models import init_chat_model
import os 


arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=4000)
MAX_STEPS = 6
llm = init_chat_model("qwen2.5:3b", model_provider="ollama", base_url="127.0.0.1:11500")

print(llm.invoke("what's your name").content)
# llm = ChatAnthropic(
#     model="claude-sonnet-5-5",
#     default_headers={
#         "anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"],
#     },
# )
# Define tools 
@tool
def archiv_search(query: str):
    """Search arXiv for academic papers. Returns titles, authors, dates and abstracts."""
    return arxiv.run(query)    
 
# @tool
# def web_search(query: str):
#     """Search the web for recent articles, blog posts, and general information."""
#     return str(tavily.invoke({"query": query}))

tools = [archiv_search]   
tools_by_name = {t.name:t for t in tools}
# Define llm with tools
llm_with_tools = llm.bind_tools(tools)


# Schema for structured output to use in planning
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the Plan for litreature review of the given topic.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)

# Define workers with tools for the orchestrator

@task 
def research_section(section: Section, topic: str):
    """Worker: researches one section using tools, then writes it."""
    messages = [
        SystemMessage(content=(
            "You are a research assistant writing one section of a literature review. "
            "Use the search tools to find relevant papers and sources, then write the section "
            "in markdown. Cite sources inline (title + year + link). Do not invent citations."
        )),
        HumanMessage(content=(
            f"Overall topic: {topic}\n"
            f"Section: {section.name}\n"
            f"Scope: {section.description}"
        )),
    ]
    
    for _ in range(MAX_STEPS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # check if the llm finish searching 
        if not response.tool_calls:
            return f"## {section.name}\n\n{response.content}"
        
        # track used tools
        for tool_call in response.tool_calls:
            tool_msg = tools_by_name[tool_call["name"]].invoke(tool_call)  # returns a ToolMessage
            messages.append(tool_msg)
            
    # Hit the step limit: force a final write-up without tools
    final = llm.invoke(messages + [HumanMessage(content="Now write the section using what you found.")])
    return f"## {section.name}\n\n{final.content}"

@task
def synthesizer(topic: str, completed_sections: List[str]) -> str:
    body = "\n\n---\n\n".join(completed_sections)
    return f"# Literature Review: {topic}\n\n{body}"

@task
def orchestrator(topic: str):
    """Orchestrator that generates a plan for the litreature review"""
    # Generate queries (plan for the llm )
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan to conduct a litreature review for the given topic."),
            HumanMessage(content=f"Here is the research topic: {topic}"),
        ]
    )

    return report_sections.sections





@entrypoint()
def orchestrator_worker(topic: str):
    sections = orchestrator(topic).result()
    
    # Call workers
    futures = [research_section(section, topic) for section in sections ]
    completed = [f.result() for f in futures]
    # return response
    return synthesizer(topic, completed).result()

# Invoke
report = orchestrator_worker.invoke("Create a report on LLM medical imaging in echocardiography")
from IPython.display import Markdown
Markdown(report)