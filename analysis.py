import pandas as pd
import os
import re
from rdflib import ConjunctiveGraph
from SPARQLWrapper import SPARQLWrapper, JSON
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')

# Directory containing the CSV files
input_directory = "./"  
output_directory = "./processed_results"
os.makedirs(output_directory, exist_ok=True)

def extract_sparql(query_text):
    """
    Extract SPARQL queries from the text. Handles multiple queries in the same block.
    """
    if pd.isnull(query_text) or not isinstance(query_text, str):
        return None

    # Match SPARQL blocks enclosed in ```sparql
    queries = re.findall(r"```sparql(.*?)```", query_text, re.DOTALL)
    
    # If no ```sparql delimiter is found, look for common SELECT/ASK/CONSTRUCT patterns
    if not queries:
        queries = re.findall(
            r"(?:(?:PREFIX|SELECT|ASK|CONSTRUCT).*?WHERE\s*{.*?})",
            query_text,
            re.DOTALL | re.IGNORECASE,
        )
    
    # Clean up extracted queries (remove leading/trailing whitespace)
    queries = [query.strip() for query in queries]
    
    # Return queries as a concatenated string if multiple exist
    return "\n\n".join(queries) if queries else None

# Function to validate SPARQL syntax
def validate_sparql_syntax(query):
    """Validate the syntax of a SPARQL query."""
    try:
        g = ConjunctiveGraph()
        g.query(query)
        return True
    except Exception:
        return False

# Function to execute SPARQL query
def check_query_execution(query, endpoint="https://dbpedia.org/sparql"):
    """Check if the query executes successfully on the endpoint."""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        sparql.query().convert()
        return True
    except Exception:
        return False

# Function to calculate BLEU score
def calculate_bleu_score(reference_query, generated_query):
    """Calculate BLEU score comparffing reference and generated queries."""
    reference = reference_query.split()
    candidate = generated_query.split()
    return sentence_bleu([reference], candidate)

# Function to calculate Macro-F1 score
def calculate_macro_f1(reference_query, generated_query):
    """Calculate Macro-F1 score comparing reference and generated queries."""
    reference_set = set(reference_query.split())
    generated_set = set(generated_query.split())
    all_items = list(reference_set.union(generated_set))
    reference_binary = [1 if item in reference_set else 0 for item in all_items]
    generated_binary = [1 if item in generated_set else 0 for item in all_items]
    return f1_score(reference_binary, generated_binary, average="macro")

# Function to process a single file
def process_file(file_path):
    data = pd.read_csv(file_path)
    data['parsed_generated_query'] = data['generated_query'].apply(extract_sparql)
    data['is_syntax_valid'] = data['parsed_generated_query'].apply(
        lambda query: validate_sparql_syntax(query) if query else False
    )
    data['is_execution_valid'] = data['parsed_generated_query'].apply(
        lambda query: check_query_execution(query) if query else False
    )
    data['bleu_score'] = data.apply(
        lambda row: calculate_bleu_score(row['reference_query'], row['parsed_generated_query'])
        if row['parsed_generated_query'] and row['reference_query']
        else None,
        axis=1,
    )
    data['macro_f1'] = data.apply(
        lambda row: calculate_macro_f1(row['reference_query'], row['parsed_generated_query'])
        if row['parsed_generated_query'] and row['reference_query']
        else None,
        axis=1,
    )
    return data

# Aggregate results for all files
aggregate_stats = []

for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):
        print(f"Processing file: {file_name}")
        file_path = os.path.join(input_directory, file_name)
        processed_data = process_file(file_path)
        
        # Save processed data to a new file
        output_file = os.path.join(output_directory, f"processed_{file_name}")
        processed_data.to_csv(output_file, index=False)
        
        # Compute aggregate statistics for each quantization level
        for quantization_level, quant_data in processed_data.groupby("quantization"):
            model_name = quant_data['model'].iloc[0]  # Extract model name from the "model" column
            total_queries = len(quant_data)
            syntax_valid_count = quant_data['is_syntax_valid'].sum()
            execution_valid_count = quant_data['is_execution_valid'].sum()
            average_bleu = quant_data['bleu_score'].mean()
            average_f1 = quant_data['macro_f1'].mean()

            aggregate_stats.append({
                "Model": model_name,
                "Quantization": quantization_level,
                "Total Queries": total_queries,
                "Syntax Valid Queries": syntax_valid_count,
                "Execution Valid Queries": execution_valid_count,
                "Average BLEU": average_bleu,
                "Average Macro-F1": average_f1,
            })

# Save aggregate statistics to a summary CSV
summary_df = pd.DataFrame(aggregate_stats)
summary_output_path = os.path.join(output_directory, "aggregate_stats_with_quantization.csv")
summary_df.to_csv(summary_output_path, index=False)

# Display summary
print("\nAggregate Statistics:")
print(summary_df)
print(f"\nSummary report saved as '{summary_output_path}'")
