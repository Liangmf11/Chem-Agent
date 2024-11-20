import requests
from langchain.tools import tool

class Name2SMILES:
    """
    Convert a molecule name to its SMILES representation using external APIs.
    """
    def __init__(self, chemspace_api_key=None):
        self.pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
        # self.chemspace_api_key = chemspace_api_key
        # self.chemspace_url = "https://chem-space.com/api/v1/convert"

    def query_pubchem(self, name):
        """
        Query PubChem for the SMILES representation of a molecule.
        """
        try:
            response = requests.get(self.pubchem_url.format(name))
            response.raise_for_status()
            data = response.json()
            smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            return smiles
        except Exception as e:
            return f"PubChem query failed: {e}"
        
    def run(self, query):
        """
        Main function to get SMILES from PubChem and fallback to ChemSpace if needed.
        """
        smiles = self.query_pubchem(query)
        # if "PubChem query failed" in smiles:
        #     smiles = self.query_chemspace(query)
        return smiles