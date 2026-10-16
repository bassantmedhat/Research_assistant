import asyncio, sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.func import entrypoint, task
from langchain.chat_models import init_chat_model
import os 


MAX_STEPS = 6
MAX_REVISIONS = 2
llm = init_chat_model("qwen2.5:3b", model_provider="ollama", base_url="127.0.0.1:11500")

print(llm.invoke("what's your name").content)


# Define mcp server client 
mcp_client = MultiServerMCPClient({
    "research": {
        "command": sys.executable,   # same Python env 
        "args": [str(Path(__file__).parent / "tools_server.py")],
        "transport": "stdio",
    },
})
 
tools_by_name, llm_with_tools = {}, None   # filled in main() once tools are loaded



# ---------- Schemas ----------
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
    
class Verdict(BaseModel):
    approved: bool = Field(description="True only if every claim is supported by the evidence.")
    issues: List[str] = Field(default_factory=list,
        description="Concrete problems: citations not in the evidence, unsupported claims, scope gaps.")


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)
verifier = llm.with_structured_output(Verdict)


# ---------- Agents ----------

@task 
async def plan(topic: str):
    """Planner / orchestrator: splits the topic into sections."""
    result = planner.ainvoke([
        SystemMessage(content="Generate a plan (list of sections) for a literature review on the given topic."),
        HumanMessage(content=f"Research topic: {topic}"),

    ])
    return result.sections


@task 
async def junior_researcher(section: Section, topic: str, feedback: Optional[str] = None, previous: Optional[str] = None):
    """Worker: tool-calling loop, returns the draft plus the raw evidence it retrieved."""
    prompt = f"Overall topic: {topic}\nSection: {section.name}\nScope: {section.description}"
    if feedback:
        prompt += f"\n\nYour previous draft:\n{previous}\n\nA senior reviewer rejected it:\n{feedback}\nFix these issues."
    messages = [
        SystemMessage(content=(
            "You are a junior research assistant writing one section of a literature review. "
            "First call recall_notes, then arxiv_search. Save key findings with save_note. "
            "Write the section in markdown, citing sources inline (title + year + link). "
            "Only cite papers that appear in your tool results.")),
        HumanMessage(content=prompt),
    ]
    evidence = []
    for _ in range(MAX_STEPS):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"text": response.content, "evidence": evidence}
        for call in response.tool_calls:
            tool_msg = await tools_by_name[call["name"]].ainvoke(call)
            messages.append(tool_msg)
            if call["name"] == "arxiv_search":
                evidence.append(str(tool_msg.content))

    final = await llm.ainvoke(messages + [HumanMessage(content="Now write the section using what you found.")])
    return {"text": final.content, "evidence": evidence}

@task
async def senior_verifier(section: Section, draft: dict) -> dict:
    """Checks the draft against the evidence the junior actually retrieved."""
    if not draft["evidence"]:
        return {"approved": False, "issues": ["No sources retrieved. Search arXiv before writing."]}
    evidence = "\n\n".join(draft["evidence"])[:12000]
    verdict = await verifier.ainvoke([
        SystemMessage(content=(
            "You are a senior researcher. Verify the draft: every cited paper must appear in the EVIDENCE, "
            "and every claim must be supported by it. Also flag parts of the scope that are missing.")),
        HumanMessage(content=f"Scope: {section.description}\n\nDRAFT:\n{draft['text']}\n\nEVIDENCE:\n{evidence}"),
    ])
    return verdict.model_dump()


@task
async def synthesizer(topic: str, sections: List[str]) -> str:
    """Merges approved sections and writes an introduction and conclusion."""
    body = "\n\n".join(sections)
    framing = await llm.ainvoke([
        SystemMessage(content="Write a short introduction and a conclusion for this literature review. "
                              "Do not add new citations."),
        HumanMessage(content=f"Topic: {topic}\n\n{body}"),
    ])
    return f"# Literature Review: {topic}\n\n{framing.content}\n\n---\n\n{body}"



# ---------- Pipeline ----------
async def research_with_review(section: Section, topic: str) -> str:
    draft = await junior_researcher(section, topic)
    for attempt in range(MAX_REVISIONS + 1):
        verdict = await senior_verifier(section, draft)
        if verdict["approved"] or attempt == MAX_REVISIONS:
            break
        draft = await junior_researcher(section, topic,
                                        feedback="\n".join(verdict["issues"]), previous=draft["text"])
    return f"## {section.name}\n\n{draft['text']}"


@entrypoint()
async def research_pipeline(topic: str) -> str:
    sections = await plan(topic)
    texts = await asyncio.gather(*(research_with_review(s, topic) for s in sections))
    return await synthesizer(topic, list(texts))



async def main():
    global tools_by_name, llm_with_tools
    tools = await mcp_client.get_tools()
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    report = await research_pipeline.ainvoke("LLM medical imaging in echocardiography")
    Path("report.md").write_text(report)

if __name__ == "__main__":
    asyncio.run(main())