# Chem-Agent
This repository provides a streamlined implementation of ChemCrow tools, leveraging the LangChain framework to enable effective chemical informatics. It supports various tools, agent types, and large language models (LLMs), allowing for flexible and efficient workflows.

## Features
- Supported Tools
  - Web Search: Enables retrieval of up-to-date chemical and scientific information.
  - Name2SMILES: Converts chemical names into their corresponding SMILES notation.
  - SMILES2Weight: Computes molecular weight from SMILES strings.
  - Custom Extensions: The implementation allows adding additional tools as needed.
- Supported Agent Types
  - Zero-Shot: Solves tasks directly without external reasoning or context augmentation.
  - Conversational: Engages in interactive dialogues to answer queries with contextual awareness.
  - Self Ask With Search: Employs a step-by-step reasoning process, integrating web search when necessary.
  - Custom Agents: Allows developers to define tailored agent behaviors for specific applications.
- Supported LLMs
  - OpenAI Models: Includes models such as GPT-4o and GPT-4 for general-purpose tasks.
  - TongyiQwen Models: Supports Alibaba's TongyiQwen series.
  - HuggingFace Models: Works with open-source transformer models like BERT, T5, and Chem-specific pretrained models.

## Usage
Configure your api keys in `.env` before testing the code.

```env
OPENAI_API_KEY=<YOUR-OPENAI-API-KEY-HERE>
SERPAPI_API_KEY=<YOUR-SERP-API-KEY-HERE>
...
```

Execute the following command to run the code.
```shell
# Quick start
$ python main.py --agent 1 --llm openai

```
