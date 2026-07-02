import json
import logging
from typing import List, Dict
from datasets import load_dataset

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATASET_ID = "Nerd97/shl-catalog-dataset"
OUTPUT_FILE = "catalog.json"

def fetch_and_process_catalog() -> List[Dict[str, str]]:
    """
    Fetches the SHL catalog from Hugging Face and filters out out-of-scope Job Solutions.
    """
    logger.info(f"Fetching dataset '{DATASET_ID}' from Hugging Face...")
    
    try:
        # Load the exact dataset made for this assignment
        dataset = load_dataset(DATASET_ID, split="train")
    except Exception as e:
        logger.error(f"Network or Dataset Error. Failed to fetch data: {e}")
        return []

    catalog: List[Dict[str, str]] = []
    skipped_count = 0
    
    for row in dataset:
        category = str(row.get("category", ""))
        
        # STRICT ASSIGNMENT RULE: Skip Pre-packaged Job Solutions
        if "Pre-packaged" in category or "Job" in category:
            skipped_count += 1
            continue
            
        # Format the data exactly how your RAG engine (retrieval.py) expects it
        catalog.append({
            "name": str(row.get("name", "")).strip(),
            "url": str(row.get("url", "")).strip(),
            "test_type": str(row.get("test_type", "")).strip(),
            "desc": str(row.get("description", "")).strip()
        })
        
    logger.info(f"Processed {len(catalog)} Individual Test Solutions.")
    logger.info(f"COMPLIANCE: Skipped {skipped_count} Job Solutions.")
    
    return catalog

def save_catalog(catalog_data: List[Dict[str, str]], filepath: str) -> None:
    """Saves the processed catalog list to a JSON file."""
    if not catalog_data:
        logger.error("Catalog data is empty. Aborting file save to prevent overwriting existing data with blanks.")
        return

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(catalog_data, f, indent=4)
        logger.info(f"SUCCESS: Saved data to '{filepath}'.")
    except IOError as e:
        logger.error(f"File system error. Failed to write to '{filepath}': {e}")

def main():
    catalog_data = fetch_and_process_catalog()
    save_catalog(catalog_data, OUTPUT_FILE)

if __name__ == "__main__":
    main()