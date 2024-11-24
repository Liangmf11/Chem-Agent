import argparse
import openai
from dotenv import load_dotenv
import os

from langchain_community.llms import Tongyi
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from rdkit import Chem
from rdkit.Chem import Descriptors

from tools import Name2SMILES
from agents import CustomAgent

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define tools
@tool
def add_numbers(input_str: str) -> str:
    """Adds two numbers given as a single comma-separated string."""
    try:
        a, b = map(float, input_str.split(","))
        return str(a + b)
    except ValueError:
        return "Invalid input. Provide two numbers separated by a comma."

@tool
def name_to_smiles(query: str) -> str:
    """Convert a molecule name to its SMILES representation using PubChem and ChemSpace."""
    name_to_smiles = Name2SMILES(chemspace_api_key="your_chemspace_api_key")
    return name_to_smiles.run(query)

@tool
def smiles_to_weight(smiles: str) -> str:
    """Calculate the molecular weight of a molecule from its SMILES representation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            weight = Descriptors.MolWt(mol)
            return f"The molecular weight of {smiles} is {weight:.2f} g/mol."
        return "Invalid SMILES string."
    except Exception as e:
        return f"Error: {e}"

# Load a search tool
search_tool = load_tools(["serpapi"])[0]

self_ask_search_tool = [Tool(name="Intermediate Answer", func=search_tool.run, description="A search engine. Use this to find answers via web search.")]

tools = [
    Tool(name="Calculator", func=add_numbers.run, description="Add two numbers."),
    Tool(name="Search", func=search_tool.run, description="Search the web for information."),
    Tool(name="Name2SMILES", func=name_to_smiles.run, description="Convert molecule names to SMILES."),
    Tool(name="SMILES2Weight", func=smiles_to_weight.run, description="Calculate molecular weight from SMILES.")
]

# Initialize memory and LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Initialize the agent
custom_agent = CustomAgent(tools=tools)
custom_agent_executor = AgentExecutor(agent=custom_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Main program
def main(agent_choice):
    agent_type_mapping = {
        1: AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        2: AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        3: AgentType.REACT_DOCSTORE,
        4: AgentType.SELF_ASK_WITH_SEARCH,
        5: None  # Custom agent
    }
    selected_agent = custom_agent_executor if agent_choice == 5 else initialize_agent(
        self_ask_search_tool if agent_choice == 4 else tools,
        llm,
        agent=agent_type_mapping.get(agent_choice, AgentType.CONVERSATIONAL_REACT_DESCRIPTION),
        memory=memory if agent_choice == 2 else None,
        verbose=True
    )

    print("Hello! How can I assist you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = selected_agent.invoke({"input": user_input})
        print(f"Agent: {response['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the agent type for the chatbot.")
    parser.add_argument(
        "-a", "--agent", 
        type=int, 
        choices=[1, 2, 3, 4, 5],
        default=2,
        help="1: Zero-shot, 2: Conversational (default), 3: Docstore, 4: Self-ask, 5: Custom agent"
    )
    args = parser.parse_args()
    main(args.agent)
