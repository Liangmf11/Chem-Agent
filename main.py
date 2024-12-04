import argparse
import openai
import requests
import re
from dotenv import load_dotenv
import os
from huggingface_hub import login

from langchain.tools import tool
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
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

@tool
def fetch_uniprot_sequences(ec_number: str, output_file: str = "uniprot_sequences.fasta") -> str:
    """
    Fetches UniProt sequences for a given EC number and saves them to a FASTA file.
    Adds a check for the format of the EC number before making the API request.
    
    Args:
        ec_number (str): The EC number to search for in UniProt.
        output_file (str): The path to the file where sequences will be saved.

    Returns:
        str: A message indicating the result of the operation.
    """
    
    # Remove leading/trailing whitespaces from input
    ec_number = ec_number.strip()
    
    # Check if the EC number matches the expected format (e.g., 1.1.1.1)
    ec_number_pattern = r"^\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,2}$"
    
    if not re.match(ec_number_pattern, ec_number):
        return f"Invalid EC number format: '{ec_number}'. The correct format is 'X.X.X.X', where X is a number."

    # If EC number format is valid, proceed with the UniProt request
    query_url = f"https://rest.uniprot.org/uniprotkb/search?query=ec:{ec_number}&format=fasta"
    
    # Perform the GET request
    response = requests.get(query_url)
    
    if response.status_code == 200:
        # Save the response text (FASTA data) to the specified file
        with open(output_file, "w") as f:
            f.write(response.text)
        return f"UniProt sequences for EC {ec_number} saved to {output_file}."
    else:
        # If there is an error, return a message indicating the failure
        return f"Failed to fetch UniProt sequences for EC {ec_number}. Status code: {response.status_code}. Error: {response.text}"

# Load a search tool
search_tool = load_tools(["serpapi"])[0]

self_ask_search_tool = [Tool(name="Intermediate Answer", func=search_tool.run, description="A search engine. Use this to find answers via web search.")]

tools = [
    Tool(name="Calculator", func=add_numbers.run, description="Add two numbers."),
    # Tool(name="Search", func=search_tool.run, description="Search the web for information."),
    Tool(name="Name2SMILES", func=name_to_smiles.run, description="Convert molecule names to SMILES."),
    Tool(name="SMILES2Weight", func=smiles_to_weight.run, description="Calculate molecular weight from SMILES."),
    Tool(name="Fetch UniProt Sequences", func=fetch_uniprot_sequences.run, description="Fetch UniProt sequences for a given EC number.")
]

# Initialize memory and LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
def initialize_llm(model_name: str, **kwargs):
    if model_name.lower() == "openai":
        return ChatOpenAI(model=kwargs.get("model", "gpt-4o-mini"), temperature=kwargs.get("temperature", 0.1))
    elif model_name.lower() == "tongyi":
        return ChatOpenAI(temperature=kwargs.get("temperature", 0.1), 
                          openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
                          openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                          model="qwen-max") 
    elif model_name.lower() == "hf":
        return ChatOpenAI(temperature=kwargs.get("temperature", 0.1), 
                          openai_api_key=os.getenv("HUGGINGFACE_API_KEY"),
                          openai_api_base="https://api-inference.huggingface.co/v1/", 
                          model="Qwen/Qwen2.5-Coder-32B-Instruct") 
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
# Initialize the agent
custom_agent = CustomAgent(tools=tools)
custom_agent_executor = AgentExecutor(agent=custom_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Main program
def main(agent_choice, llm_choice="openai"):
    llm = initialize_llm(llm_choice, model="gpt-4o-mini", temperature=0.1)
    
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
    parser = argparse.ArgumentParser(description="Choose the agent type and LLM for the chatbot.")
    parser.add_argument(
        "-a", "--agent", 
        type=int, 
        choices=[1, 2, 3, 4, 5],
        default=2,
        help="1: Zero-shot, 2: Conversational (default), 3: Docstore, 4: Self-ask, 5: Custom agent"
    )
    parser.add_argument(
        "-l", "--llm", 
        type=str, 
        choices=["openai", "tongyi", "hf"],
        default="openai",
        help="Specify the LLM to use (default: openai)."
    )
    args = parser.parse_args()
    main(args.agent, args.llm)
