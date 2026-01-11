# new simple code 

# THIS IS THE CODE USING THE 5P'S MODEL 

import networkx as nx
from transformers import pipeline
import torch
import pandas as pd
#import re
import regex as re
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import requests
import ast
import matplotlib
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pickle

from huggingface_hub import login

current_dir = Path(__file__).parent  # Assumes your script is in the current folder

# Navigate to sibling directory (adjust "deepseek_model_folder" to your actual folder name)
model_id2= current_dir.parent.parent / "models" / "deepseek-llm-7b-chat"

model_id = current_dir.parent.parent / "models" / "Meta-Llama-3-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model1 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

def preprocess_csv(file_path):
    """
    Reads conversation data from a CSV file. The CSV is assumed to have two columns:
    'speaker_id' and 'utterance'. It concatenates the conversation by prepending 
    the speaker_id before each utterance.
    """
    df = pd.read_csv(file_path)

    conversation = ""

    for index, row in df.iterrows():
        speaker_raw = row.get('speaker_id', '')
        utterance_raw = row.get('utterance', '')

        # Handle NaNs or unexpected types
        speaker = str(speaker_raw).strip() if not pd.isna(speaker_raw) else ''
        utterance = str(utterance_raw).strip() if not pd.isna(utterance_raw) else ''

        if speaker == 'SPEAKER_01':
            conversation += f"Therapist: {utterance}\n"
        else:
            conversation += f"Patient: {utterance}\n"

    return conversation


class LLMGraphGenerator:
    def __init__(self, conversation_file, max_iterations=10, model_id = model_id, model_id2 = model_id2):
        """
        Initialize the LLaMA model and tokenizer from Hugging Face. You can specify a different LLaMA model if desired.
        """
        # Load the model and tokenizer from Hugging Face
        self.model1 = model1
        #self.model2 = model2
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        #self.tokenizer2 = AutoTokenizer.from_pretrained(model_id2)
        self.max_iterations = max_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1.to(self.device)
        #self.model2.to(self.device) - not to be uncommented
        self.convo = preprocess_csv(conversation_file)
        self.transcript = self.convo
        self.presenting_problems = []
        self.prec_factors = []
        self.predisposing_factors = []
        self.perpetuating_factors = []
        self.edges = []  # stores the graph edges

    def parse_nodes(self, response):
        """
        Robustly extracts factors from response and returns them as four separate lists.
        
        Args:
            response: Raw string response from the model
            
        Returns:
            Tuple containing (predisposing_factors, precipitating_factors, 
            presenting_problems, perpetuating_factors)
            Returns empty lists if parsing fails
        """
        # Initialize empty lists
        predisposing = []
        precipitating = []
        presenting = []
        perpetuating = []
        
        # Preprocessing steps
        cleaned_response = response.strip()
        
        # Strategy: Work backwards to find the last valid JSON object
        stack = []
        json_candidates = []
        in_string = False
        escape = False
        
        # Reverse scan to find balanced braces
        for i, char in enumerate(reversed(cleaned_response)):
            if char == '"' and not escape:
                in_string = not in_string
            elif char == '\\':
                escape = not escape
            else:
                escape = False
                
            if not in_string:
                if char == '}':
                    stack.append(i)
                elif char == '{':
                    if stack:
                        start_pos = stack.pop()
                        if not stack:  # Found balanced braces
                            # Convert reversed positions to normal indices
                            start = len(cleaned_response) - start_pos - 1
                            end = len(cleaned_response) - i
                            candidate = cleaned_response[start:end]
                            json_candidates.append(candidate)
        
        # Try parsing candidates from most recent to oldest
        for candidate in reversed(json_candidates):
            try:
                # First try strict parsing
                data = json.loads(candidate)
                # Extract factors if they exist
                predisposing = data.get("predisposing_factors", [])
                precipitating = data.get("precipitating_factors", [])
                presenting = data.get("presenting_problems", [])
                perpetuating = data.get("perpetuating_factors", [])
                return (predisposing, precipitating, presenting, perpetuating)
            except json.JSONDecodeError:
                # Fallback: try fixing common issues
                try:
                    # Handle trailing commas
                    fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
                    # Handle unquoted keys
                    fixed = re.sub(r'([{,])\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', fixed)
                    data = json.loads(fixed)
                    # Extract factors if they exist
                    predisposing = data.get("predisposing_factors", [])
                    precipitating = data.get("precipitating_factors", [])
                    presenting = data.get("presenting_problems", [])
                    perpetuating = data.get("perpetuating_factors", [])
                    return (predisposing, precipitating, presenting, perpetuating)
                except json.JSONDecodeError:
                    continue
        
        # Final fallback: try the whole string as JSON
        try:
            data = json.loads(cleaned_response)
            # Extract factors if they exist
            predisposing = data.get("predisposing_factors", [])
            precipitating = data.get("precipitating_factors", [])
            presenting = data.get("presenting_problems", [])
            perpetuating = data.get("perpetuating_factors", [])
            return (predisposing, precipitating, presenting, perpetuating)
        except json.JSONDecodeError:
            pass
        
        # If all else fails, try extracting with more permissive regex
        json_matches = re.findall(r'{(?:[^{}]|(?R))*}', cleaned_response, flags=re.DOTALL)
        if json_matches:
            try:
                data = json.loads(json_matches[-1])
                # Extract factors if they exist
                predisposing = data.get("predisposing_factors", [])
                precipitating = data.get("precipitating_factors", [])
                presenting = data.get("presenting_problems", [])
                perpetuating = data.get("perpetuating_factors", [])
                return (predisposing, precipitating, presenting, perpetuating)
            except json.JSONDecodeError:
                pass
        
        # Return empty lists if no valid JSON found
        return ([], [], [], [])

    
    def identify_nodes(self, prompt):
        """
        STEP 1: Identify the immediate presenting problems that have been mentioned directly 
        """
        with open(prompt, 'r') as file:
            prompt_template = file.read()

        prompt = prompt_template.replace("{{conversation}}", self.convo)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model1.device)
        
        response = self.model1.generate(**inputs, max_new_tokens=12000)
        response = self.tokenizer.decode(response[0])
        #response = self.deep_generate(prompt, max_new_tokens=12000)

        # once we have received the response from the LLM, we need to convert it to list format
        print(response)

        self.predisposing_factors, self.prec_factors, self.presenting_problems, self.perpetuating_factors = self.parse_nodes(response)

        # all the possible nodes have been identified

    # Once all the causes have been recorded, we validate the edges and remove any that should not be there
    
    def parse_validation(self, response_text):
        """
        Parses the model's response to determine if the edge is correctly identified.

        Args:
            response_text (str): The raw output from the language model.

        Returns:
            bool: True if the edge is correct, False if incorrect.

        Raises:
            ValueError: If no valid 'edge_correct' JSON-like object is found or parsed.
        """
        # Find all JSON-like blocks that contain "edge_correct"
        json_blocks = list(re.finditer(
            r'\{[^{}]*"edge_correct"\s*:\s*\[\s*"(?P<val>true|false)"\s*\][^{}]*\}',
            response_text,
            re.IGNORECASE
        ))

        if not json_blocks:
            raise ValueError("No valid 'edge_correct' object found in the response.")

        # Try each block starting from the last one (assuming it's the most complete)
        for match in reversed(json_blocks):
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
                value_list = parsed.get("edge_correct")
                if isinstance(value_list, list) and len(value_list) > 0:
                    value = value_list[0]
                    if isinstance(value, str):
                        value_lower = value.strip().lower()
                        if value_lower == "true":
                            return True
                        elif value_lower == "false":
                            return False
                # If parsing but format unexpected, continue to next match
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                continue  # Skip to next candidate

        # If loop finishes without returning
        raise ValueError("Valid 'edge_correct' block found but none could be parsed properly.")

    def validate_edge(self, edge, prompt_file):
        """
        Uses LLaMA to validate a causal relationship between a symptom and a disease.
        """
        node1, node2 = edge

        # Read validation prompt from the .txt file
        with open(prompt_file, 'r') as file:
            prompt_template = file.read()

        # Insert edge details into the prompt template
        prompt = prompt_template.replace("{{node1}}", node1).replace("{{node2}}", node2).replace("{{context}}", self.convo)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model1.device)
        
        response = self.model1.generate(**inputs, max_new_tokens=5000)
        response_text = self.tokenizer.decode(response[0])

        print("Feedback received:")
        print(response_text)

        is_valid = self.parse_validation(response_text)

        return is_valid
    

    def prune_edges(self, prompt_file):
        """
        Validates and prunes hypothetical inter-category edges using the given LLM-based validator.

        Args:
            validate_edge (function): A function that takes (node1, node2, context) and returns a bool.
        """
        category_lists = [
            self.presenting_problems,
            self.prec_factors,
            self.predisposing_factors,
            self.perpetuating_factors
        ]

        # Generate inter-list edge candidates
        for i, source_list in enumerate(category_lists):
            for j, target_list in enumerate(category_lists):
                if i == j:
                    continue  # skip same-category edges
                for source_node in source_list:
                    for target_node in target_list:
                        try:
                            if self.validate_edge((source_node, target_node), prompt_file):
                                self.edges.append((source_node, target_node))
                        except Exception as e:
                            print(f"Validation error for edge ({source_node} -> {target_node}): {e}")
    
    def make_graph_old(self, output_file="graph.png"):
        G = nx.DiGraph()
        edges = self.edges
        G.add_edges_from(edges)
        
        # Identify nodes with no outgoing edges
        all_nodes = set(G.nodes)
        nodes_with_outgoing = {u for u, v in G.edges}
        leaf_nodes = all_nodes - nodes_with_outgoing
        
        # Add the root node 'PATIENT'
        G.add_node("PATIENT")
        
        # Connect all leaf nodes to 'PATIENT'
        for node in leaf_nodes:
            G.add_edge(node, "PATIENT")
        
        # Draw and save the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
        plt.savefig(output_file)
        plt.close()

        print(f"Plot saved as {output_file}.")
        
        return G
    
    def make_graph(self, output_file="graph"):

        G = nx.DiGraph()
        edges = self.edges
        G.add_edges_from(edges)

        # Identify nodes with no outgoing edges
        all_nodes = set(G.nodes)
        nodes_with_outgoing = {u for u, v in G.edges}
        leaf_nodes = all_nodes - nodes_with_outgoing

        # Add the root node 'PATIENT'
        G.add_node("PATIENT")

        # Connect all leaf nodes to 'PATIENT'
        for node in leaf_nodes:
            G.add_edge(node, "PATIENT")

        # Save graph object
        graph_path = f"{output_file}_graph.gpickle"
        #nx.write_gpickle(G, graph_path)
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print(f"Graph saved as {graph_path}")

        # Define layout types and their corresponding functions
        layout_funcs = {
            "spring": nx.spring_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "shell": lambda g: nx.shell_layout(g, nlist=[
                self.predisposing_factors,
                self.prec_factors,
                self.perpetuating_factors,
                self.presenting_problems,
                ["PATIENT"]
            ])
        }

        # Create and save plots for each layout
        for layout_name, layout_func in layout_funcs.items():
            plt.figure(figsize=(8, 6))
            try:
                pos = layout_func(G)
                nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
                layout_file = f"{output_file}_{layout_name}.png"
                plt.savefig(layout_file)
                print(f"Plot saved as {layout_file}")
            except Exception as e:
                print(f"Failed to generate {layout_name} layout: {e}")
            finally:
                plt.close()

        return G

    def run(self, presenting_problems_prompt_file, precipitating_causes_prompt_file, causes_prompt_file, validation_prompt_file, n, output_file):
        """
        The main function that executes the graph generation and refinement process.
        """
        # Step 1: Identify presenting problems
        print("identifying problems")
        self.identify_nodes(presenting_problems_prompt_file)
        print("problems identified: ", self.presenting_problems)

        # Step 4: Prune edges 
        print("pruning edges")
        self.prune_edges(validation_prompt_file)

        # Step 5: Generate Graph
        print("generating graph")
        final_graph = self.make_graph(output_file)

        return final_graph

# Execution

"""
csv_file_path = "selected_sessions/Group_2/4.csv"  # Path to your CSV file
#presenting_problems_prompt_file = "prompts_5p/all_problems.txt"
#precipitating_causes_prompt_file = "prompts_5p/precipitating_causes_prompt.txt"
#causes_prompt_file = "prompts_5p/causes_prompt.txt"
output_file = "graph_4.png"
#validation_prompt_file = "prompts_5p/validation_prompt.txt" # Path to your .txt file for validation prompt


graph_model = LLMGraphGenerator(csv_file_path, 2)

# Run the model using the conversation data and prompt files
final_graph = graph_model.run(presenting_problems_prompt_file, precipitating_causes_prompt_file, causes_prompt_file, validation_prompt_file, 2, output_file)
"""

from pathlib import Path

# Define the folder containing your transcript CSV files
transcript_folder = Path("selected_sessions/incorrect")

# Define where to save output files
output_dir = Path("selected_sessions/corrected_output")
output_dir.mkdir(parents=True, exist_ok=True)

# Define prompt paths
presenting_problems_prompt_file = "prompts_5p/all_problems.txt"
precipitating_causes_prompt_file = "prompts_5p/precipitating_causes_prompt.txt"
causes_prompt_file = "prompts_5p/causes_prompt.txt"
validation_prompt_file = "prompts_5p/validation_prompt.txt"

# Loop through all CSV files in the folder
for csv_file_path in transcript_folder.glob("*.csv"):
    print(f"\nProcessing file: {csv_file_path.name}")

    # Define output file names
    base_name = csv_file_path.stem
    png_output = output_dir / f"{base_name}.png"
    gpickle_output = output_dir / f"{base_name}.gpickle"

    # Initialize and run the model
    graph_model = LLMGraphGenerator(csv_file_path)
    final_graph = graph_model.run(
        presenting_problems_prompt_file,
        precipitating_causes_prompt_file,
        causes_prompt_file,
        validation_prompt_file,
        n=2,
        output_file=png_output
    )

    # Save NetworkX graph
    #nx.write_gpickle(final_graph, gpickle_output)

    """
    with open(gpickle_output, "wb") as f:
        pickle.dump(final_graph, f)
    """

    #print(f"Graph saved as PNG and GPickle for: {base_name}")
