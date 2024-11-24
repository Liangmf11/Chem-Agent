from typing import List, Dict
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent import BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI

from pydantic import Field


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define the Custom Agent
class CustomAgent(BaseSingleActionAgent):
    """Custom agent that decides whether to call a tool or respond directly."""

    tools: Dict[str, Tool] = Field(default_factory=dict)
    prompt: str = "You are a helpful assistant that uses tools to answer questions. Respond with a tool action or provide a direct response."

    def __init__(self, tools: List[Tool]):
        super().__init__()
        self.tools = {tool.name: tool for tool in tools}

    @property
    def input_keys(self) -> List[str]:
        """Input keys expected by the agent."""
        return ["input"]

    def plan(self, intermediate_steps: List[tuple], **kwargs) -> AgentAction:
        """Plan the agent's next step based on the input and previous steps."""
        query = kwargs["input"].strip().lower()

        if intermediate_steps:
            # Unpack the last intermediate step
            last_action, last_result = intermediate_steps[-1]
            
            if isinstance(last_action, AgentAction) and last_action.tool == "Search" and last_result:
                # Use the LLM to summarize the search results
                summary_prompt = (
                    "The following are results from a web search:\n"
                    f"{str(last_result)}\n"
                    "Please summarize this information in a concise and user-friendly way."
                )

                summary = llm.invoke(summary_prompt)
                return AgentFinish(
                    return_values={"output": summary.content},
                    log="Final response created using LLM summarization."
                )

        # Default to using the Search tool for other queries
        return AgentAction(tool="Search", tool_input=query, log=f"Using Search for {query}"
        )

    async def aplan(self, intermediate_steps: List[AgentAction], **kwargs):
        raise NotImplementedError("Async planning not implemented.")