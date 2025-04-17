from collections import defaultdict
import json
from numpy import mean
import itertools
import networkx as nx
import numpy as np
import copy
import json
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
from typing import List, Dict


# Calculate all possible topological sorts
def all_topological_sorts(graph: Dict[str, List[str]]) -> List[List[str]]:
    # If the number of nodes is greater than or equal to 10, return the list of nodes
    if len(graph["nodes"]) >= 10:
        return [graph["nodes"]]

    # Create a directed graph
    G = nx.DiGraph()

    # Save the original node names and their indices
    original_nodes = graph["nodes"]

    # Add nodes
    G.add_nodes_from(range(len(original_nodes)))

    # Add edges
    edges_with_indices = [(u, v) for u, v in graph["edges"]]
    G.add_edges_from(edges_with_indices)

    # Get all possible topological sorts
    all_sorts = list(nx.all_topological_sorts(G))

    # Convert the sorted indices back to the original node names and remove "START" and "END"
    filtered_sorts = []
    for sort in all_sorts:
        filtered_sort = [original_nodes[i] for i in sort if original_nodes[i] not in ["START", "END"]]
        filtered_sorts.append(filtered_sort)

    return filtered_sorts[:20]  # Return the first 20 sorts


# Calculate the size of the largest connected component
def largest_connected_component(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    connected_components = nx.connected_components(G)
    largest_component = max(connected_components, key=len)

    return len(largest_component)


# Match nodes
def match_node(pred_nodes: List[str], gt_nodes: List[str], sentence_model: object, match_threshold=0.6) -> dict:
    # Handle empty inputs
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return {i: -1 for i in range(len(pred_nodes))}

    len_pred = len(pred_nodes)
    len_gt = len(gt_nodes)

    # Generate embeddings and ensure correct dimensions
    node_pred_emb = sentence_model.encode(pred_nodes, convert_to_tensor=True)
    node_gt_emb = sentence_model.encode(gt_nodes, convert_to_tensor=True)

    # Ensure tensors are 2D (nodes x embedding_dim)
    if node_pred_emb.ndim == 1:
        node_pred_emb = node_pred_emb.unsqueeze(0)
    if node_gt_emb.ndim == 1:
        node_gt_emb = node_gt_emb.unsqueeze(0)

    # Align devices
    target_device = node_pred_emb.device
    node_gt_emb = node_gt_emb.to(target_device)

    # Calculate cosine similarity
    node_cosine_scores = util.cos_sim(node_pred_emb, node_gt_emb)
    node_cosine_scores = np.maximum(node_cosine_scores.cpu().numpy(), 0)

    # Build graph and compute maximum weight matching
    G = nx.Graph()

    for i in range(len_pred):
        for j in range(len_gt):
            if node_cosine_scores[i][j] > match_threshold:
                G.add_edge(i, str(j), weight=node_cosine_scores[i][j])

    max_weight_matching = nx.max_weight_matching(G)
    pred_to_gt_mapping = dict()
    for key in max_weight_matching:
        if type(key[0]) == int:
            pred_to_gt_mapping[int(key[0])] = int(key[1])
        else:
            pred_to_gt_mapping[int(key[1])] = int(key[0])

    # If a predicted node does not match any ground truth node, mark it as -1
    for i in range(len_pred):
        if i not in pred_to_gt_mapping:
            pred_to_gt_mapping[i] = -1

    return pred_to_gt_mapping


# Calculate backtracking rate
def backtracking_rate(pred_edges) -> float:
    edge_counts = defaultdict(int)
    for edge in pred_edges:
        edge_counts[edge] += 1
    backtracks = sum(count - 1 for count in edge_counts.values() if count > 1)
    return backtracks / len(pred_edges)


# Evaluate workflow graph
def t_eval_graph(pred_graph: Dict[str, List[str]], gt_graph: Dict[str, List[str]], sentence_model: object) -> Dict[str, float]:
    # Create a copy of pred_graph to avoid modifying the original data
    pred_graph_copy = {
        "nodes": pred_graph["nodes"].copy(),
        "edges": pred_graph["edges"].copy()
    }

    # Add "END" to the copy
    pred_graph_copy["nodes"].append("END")

    gt_nodes = gt_graph["nodes"]
    pred_to_gt_mapping = match_node(pred_graph_copy["nodes"], gt_nodes, sentence_model)

    if len(pred_graph_copy["nodes"]) == 0 or len(gt_nodes) == 0:
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }

    # Find matched nodes and edges
    last_pred_index = pred_graph_copy["edges"][-1][1] if pred_graph_copy["edges"] else 0
    pred_edges = pred_graph_copy["edges"]
    pred_edges.append((last_pred_index, last_pred_index + 1))

    gt_edges = gt_graph["edges"]

    matched_pred_nodes = []
    matched_gt_nodes = []
    for k, v in pred_to_gt_mapping.items():
        if v != -1:
            matched_pred_nodes.append(k)
            matched_gt_nodes.append(v)

    matched_pred_edges = []
    for pred_edge in pred_edges:
        if pred_edge[0] in matched_pred_nodes and pred_edge[1] in matched_pred_nodes:
            matched_pred_edges.append((pred_edge[0], pred_edge[1]))

    matched_gt_edges = []
    for gt_edge in gt_edges:
        if gt_edge[0] in matched_gt_nodes and gt_edge[1] in matched_gt_nodes:
            matched_gt_edges.append((gt_edge[0], gt_edge[1]))

    public_edges = []
    for mpe in matched_pred_edges:
        if mpe in matched_gt_edges:
            public_edges.append(mpe)

    # Calculate the number of connected components
    pred_connected_components = largest_connected_component(matched_pred_nodes, public_edges)
    gt_connected_components = largest_connected_component(matched_gt_nodes, public_edges)

    # Calculate precision and recall
    precision = len(public_edges) / len(pred_edges) if len(pred_edges) > 0 else 0
    recall = len(public_edges) / len(gt_edges) if len(gt_edges) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    backtracks = backtracking_rate(pred_edges)
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'backtracking_rate': backtracks
    }