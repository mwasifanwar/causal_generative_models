# core/causal_graphs.py
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import itertools

class CausalGraph:
    def __init__(self, variables: List[str], edges: List[Tuple[str, str]]):
        self.variables = variables
        self.edges = edges
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(variables)
        self.graph.add_edges_from(edges)
        
        self.adjacency_matrix = self._build_adjacency_matrix()
        self.topological_order = list(nx.topological_sort(self.graph))
        
    def _build_adjacency_matrix(self) -> np.ndarray:
        n = len(self.variables)
        adj = np.zeros((n, n))
        var_to_idx = {var: idx for idx, var in enumerate(self.variables)}
        
        for parent, child in self.edges:
            i = var_to_idx[parent]
            j = var_to_idx[child]
            adj[i, j] = 1
            
        return adj
    
    def get_parents(self, variable: str) -> List[str]:
        return list(self.graph.predecessors(variable))
    
    def get_children(self, variable: str) -> List[str]:
        return list(self.graph.successors(variable))
    
    def get_ancestors(self, variable: str) -> List[str]:
        return list(nx.ancestors(self.graph, variable))
    
    def get_descendants(self, variable: str) -> List[str]:
        return list(nx.descendants(self.graph, variable))
    
    def is_d_separated(self, x: str, y: str, z: List[str]) -> bool:
        return nx.d_separated(self.graph, {x}, {y}, set(z))
    
    def get_markov_blanket(self, variable: str) -> List[str]:
        parents = self.get_parents(variable)
        children = self.get_children(variable)
        spouses = []
        for child in children:
            spouses.extend(self.get_parents(child))
        spouses = [s for s in spouses if s != variable]
        
        return list(set(parents + children + spouses))
    
    def intervene(self, interventions: Dict[str, float]) -> 'CausalGraph':
        mutilated_graph = self.graph.copy()
        
        for variable in interventions:
            if variable in mutilated_graph:
                predecessors = list(mutilated_graph.predecessors(variable))
                for pred in predecessors:
                    mutilated_graph.remove_edge(pred, variable)
                    
        new_edges = list(mutilated_graph.edges())
        return CausalGraph(self.variables, new_edges)

class StructuralCausalModel:
    def __init__(self, graph: CausalGraph, mechanisms: Dict[str, nn.Module]):
        self.graph = graph
        self.mechanisms = mechanisms
        self.noise_distributions = {}
        
        for var in graph.variables:
            if var not in mechanisms:
                raise ValueError(f"No mechanism provided for variable {var}")
                
    def set_noise_distribution(self, variable: str, distribution: torch.distributions.Distribution):
        self.noise_distributions[variable] = distribution
        
    def sample(self, n_samples: int, noise_samples: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        samples = {}
        noise = {}
        
        for var in self.graph.topological_order:
            if noise_samples is not None and var in noise_samples:
                noise[var] = noise_samples[var]
            else:
                if var in self.noise_distributions:
                    noise[var] = self.noise_distributions[var].sample((n_samples,))
                else:
                    noise[var] = torch.randn(n_samples, 1)
                    
        for var in self.graph.topological_order:
            parents = self.graph.get_parents(var)
            parent_values = [samples[parent] for parent in parents]
            
            if parent_values:
                inputs = torch.cat(parent_values + [noise[var]], dim=-1)
            else:
                inputs = noise[var]
                
            samples[var] = self.mechanisms[var](inputs)
            
        return samples
    
    def do_intervention(self, interventions: Dict[str, torch.Tensor]) -> 'StructuralCausalModel':
        mutilated_graph = self.graph.intervene(interventions)
        new_mechanisms = self.mechanisms.copy()
        
        for var in interventions:
            def constant_mechanism(inputs):
                return interventions[var].expand(inputs.shape[0], -1)
                
            new_mechanisms[var] = constant_mechanism
            
        return StructuralCausalModel(mutilated_graph, new_mechanisms)
    
    def counterfactual(self, evidence: Dict[str, torch.Tensor], intervention: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise = self._abduct_noise(evidence)
        intervened_model = self.do_intervention(intervention)
        return intervened_model.sample(evidence[list(evidence.keys())[0]].shape[0], noise)
    
    def _abduct_noise(self, evidence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise = {}
        
        for var in self.graph.topological_order:
            if var in evidence:
                parents = self.graph.get_parents(var)
                if parents:
                    parent_values = torch.cat([evidence[parent] for parent in parents], dim=-1)
                    predicted = self.mechanisms[var](torch.cat([parent_values, torch.zeros_like(evidence[var])], dim=-1))
                    noise[var] = evidence[var] - predicted
                else:
                    noise[var] = evidence[var]
            else:
                if var in self.noise_distributions:
                    noise[var] = self.noise_distributions[var].sample((1,)).expand(evidence[list(evidence.keys())[0]].shape[0], -1)
                else:
                    noise[var] = torch.randn(evidence[list(evidence.keys())[0]].shape[0], 1)
                    
        return noise

class Intervention:
    def __init__(self, variable: str, value: torch.Tensor):
        self.variable = variable
        self.value = value
        
    def apply(self, model: StructuralCausalModel) -> StructuralCausalModel:
        return model.do_intervention({self.variable: self.value})
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        intervened_data = data.copy()
        intervened_data[self.variable] = self.value.expand(data[list(data.keys())[0]].shape[0], -1)
        return intervened_data