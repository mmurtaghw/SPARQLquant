import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb
import time
from accelerate import init_empty_weights, infer_auto_device_map

# Function to load the model and tokenizer with quantization options
def load_model(model_name, quantization='none'):
    """Load the model with the specified quantization level and device mapping."""
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)
    device_map = infer_auto_device_map(model, max_memory={0: "24GB", 1: "24GB", 2: "24GB", 3: "24GB"})
    print(f"Using device map: {device_map}")

    if quantization == '8bit':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True
        )
    elif quantization == '4bit':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    return model, tokenizer

# Function to generate SPARQL query
def generate_sparql(model, tokenizer, input_text):
    """Generate SPARQL query and log time taken."""
    print("Tokenising input")
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=400)
    end_time = time.time()
    print("Generating SPARQL")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Finished Decode")
    return generated_text, end_time - start_time

# Load the QALD-9 dataset
with open('data/QALD_9/data/qald_9_test_dbpedia.json', 'r') as file:
    qald_data = json.load(file)

# Number of cases to process (set to None to process all cases)
num_cases = None 

# Prepare results storage
results = []

# Experiment parameters
models = ["google/gemma-2-2b-it"] 
quantization_levels = ["8bit, 4bit, none"]

# Iterate over each model and quantization level first
for model_name in models:
    for quant in quantization_levels:
        print(f"\nLoading model {model_name} with {quant} quantization...")
        model, tokenizer = load_model(model_name, quantization=quant)

        # Process each question with the loaded model and quantization
        questions_to_process = qald_data["questions"][:num_cases] if num_cases is not None else qald_data["questions"]
        for question_entry in questions_to_process:
            question_id = question_entry["id"]
            question = next((q["string"] for q in question_entry["question"]), None)
            reference_query = question_entry["query"]["sparql"]

            if question:
                input_text = (
                    "Generate a SPARQL query for the input question for the DBpedia Knowledge Graph. "
                    "Ensure that the query uses proper SPARQL syntax, includes prefixes, and retrieves unique results. "
                    "Step-by-step: Identify relevant properties and structure the query accordingly.\n"
                    "Output only the SPARQL query\n"
                    f"Question: {question}\n"
                )

                print(f"Processing question ID {question_id} with model {model_name} at {quant} quantization.")
                generated_output, time_taken = generate_sparql(model, tokenizer, input_text)

                # Store results
                results.append({
                    "question_id": question_id,
                    "question": question,
                    "model": model_name,
                    "quantization": quant,
                    "generated_query": generated_output,
                    "reference_query": reference_query,
                    "time_taken": time_taken
                })

                print(f"Time taken: {time_taken}s\nGenerated Query: {generated_output}\nReference Query: {reference_query}\n")

        # Free up memory after processing all questions with this model and quantization
        del model, tokenizer
        torch.cuda.empty_cache()

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
print("\nSummary Report:")
print(results_df)

# Save the report as a CSV file
results_df.to_csv("experiment_report.csv", index=False)
print("\nExperiment report saved as 'qald_9_experiment_report_Gemma2b_8bit.csv'")
