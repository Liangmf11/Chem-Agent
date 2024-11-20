import openai
from dotenv import load_dotenv
import os

from langchain_community.llms import Tongyi
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from rdkit import Chem
from rdkit.Chem import Descriptors

from tools import Name2SMILES

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a simple addition tool
@tool
def add_numbers(input_str: str) -> str:
    """Adds two numbers given as a single comma-separated string."""
    try:
        a, b = map(float, input_str.split(","))
        result = a + b
        return str(result)
    except ValueError:
        return "Invalid input format. Please provide two numbers separated by a comma."

# Tool 1: Name2SMILES
# @tool
# def name_to_smiles(molecule_name: str) -> str:
#     """
#     Converts a molecule name to its SMILES representation.
#     Uses RDKit as a basic example. In a real-world scenario, 
#     this could query ChemSpider, PubChem, or OPSIN APIs.
#     """
#     try:
#         smiles = {
#             "water": "O",
#             "ethanol": "CCO",
#             "benzene": "C1=CC=CC=C1"
#         }.get(molecule_name.lower(), None)
#         if smiles:
#             return f"The SMILES representation of {molecule_name} is {smiles}."
#         else:
#             return f"Unable to find the SMILES for {molecule_name}. Please provide another name."
#     except Exception as e:
#         return f"An error occurred: {e}"


# Tool 1: Name2SMILES
@tool
def name_to_smiles(query: str) -> str:
    """
    Convert a molecule name to its SMILES representation using PubChem and ChemSpace.
    """
    name_to_smiles = Name2SMILES(chemspace_api_key="your_chemspace_api_key")
    result = name_to_smiles.run(query)
    return result

# Tool 2: SMILES2Weight
@tool
def smiles_to_weight(smiles: str) -> str:
    """
    Calculates the molecular weight of a molecule given its SMILES representation.
    Uses RDKit for calculations.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            weight = Descriptors.MolWt(mol)
            return f"The molecular weight of the molecule with SMILES {smiles} is {weight:.2f} g/mol."
        else:
            return "Invalid SMILES string provided."
    except Exception as e:
        return f"An error occurred: {e}"
    

# Load search tools
search_tool = load_tools(["serpapi"])[0]

# Define tools
tools = [
    Tool(
        name="Calculator",
        func=add_numbers.run,
        description="Use this to add two numbers. Input should be two numbers separated by a comma."
    ),
    Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool for web searches."
    ),
    Tool(
        name="Name2SMILES",
        func=name_to_smiles.run,
        description="Convert a molecule name to its SMILES representation."
    ),
    Tool(
        name="SMILES2Weight",
        func=smiles_to_weight.run,
        description="Calculate the molecular weight from a SMILES string."
    ),
]

# Initialize memory for multi-round conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize a llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# llm = Tongyi(temperature=0.1)

# Create the agent with memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",  # Conversational agent type
    memory=memory,
    verbose=True
)


if __name__ == "__main__":
    print("Hello, how can I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break
        response = agent.run(user_input)
        print(f"Agent: {response}")
