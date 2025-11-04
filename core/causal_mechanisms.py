# core/causal_mechanisms.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Callable

class CausalMechanism(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AdditiveNoiseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, noise_std: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise_std = noise_std
        
        self.mechanism = CausalMechanism(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        deterministic = self.mechanism(x)
        
        if noise is None:
            noise = torch.randn_like(deterministic) * self.noise_std
            
        return deterministic + noise
    
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        deterministic = self.mechanism(x)
        noise = y - deterministic
        log_prob = -0.5 * (noise ** 2) / (self.noise_std ** 2) - torch.log(torch.tensor(self.noise_std * np.sqrt(2 * np.pi)))
        return log_prob.mean()

class NeuralCausalModel(nn.Module):
    def __init__(self, variables: List[str], graph_edges: List[Tuple[str, str]], 
                 mechanism_dims: Dict[str, int], hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.variables = variables
        self.graph_edges = graph_edges
        self.graph = CausalGraph(variables, graph_edges)
        
        self.mechanisms = nn.ModuleDict()
        self.noise_stds = nn.ParameterDict()
        
        for var in variables:
            parents = self.graph.get_parents(var)
            input_dim = len(parents) + 1
            output_dim = mechanism_dims.get(var, 1)
            
            self.mechanisms[var] = CausalMechanism(input_dim, output_dim, hidden_dims)
            self.noise_stds[var] = nn.Parameter(torch.tensor(0.1))
            
    def forward(self, interventions: Optional[Dict[str, torch.Tensor]] = None, 
                n_samples: int = 1) -> Dict[str, torch.Tensor]:
        samples = {}
        noise = {}
        
        for var in self.variables:
            if interventions is not None and var in interventions:
                samples[var] = interventions[var]
            else:
                noise[var] = torch.randn(n_samples, 1) * torch.abs(self.noise_stds[var])
                
        for var in self.graph.topological_order:
            if var in samples:
                continue
                
            parents = self.graph.get_parents(var)
            if parents:
                parent_inputs = torch.cat([samples[parent] for parent in parents], dim=-1)
                mechanism_input = torch.cat([parent_inputs, noise[var]], dim=-1)
            else:
                mechanism_input = noise[var]
                
            samples[var] = self.mechanisms[var](mechanism_input)
            
        return samples
    
    def log_prob(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        log_prob = 0.0
        
        for var in self.graph.topological_order:
            parents = self.graph.get_parents(var)
            
            if parents:
                parent_inputs = torch.cat([data[parent] for parent in parents], dim=-1)
                deterministic = self.mechanisms[var](torch.cat([parent_inputs, torch.zeros_like(data[var])], dim=-1))
            else:
                deterministic = self.mechanisms[var](torch.zeros_like(data[var]))
                
            noise = data[var] - deterministic
            var_log_prob = -0.5 * (noise ** 2) / (torch.abs(self.noise_stds[var]) ** 2) - torch.log(torch.abs(self.noise_stds[var]))
            log_prob += var_log_prob.mean()
            
        return log_prob
    
    def causal_effect(self, treatment: str, outcome: str, 
                     treatment_value: float, reference_value: float, 
                     n_samples: int = 1000) -> torch.Tensor:
        treatment_intervention = torch.tensor([[treatment_value]], dtype=torch.float32)
        reference_intervention = torch.tensor([[reference_value]], dtype=torch.float32)
        
        treated_data = self.forward({treatment: treatment_intervention}, n_samples)
        reference_data = self.forward({treatment: reference_intervention}, n_samples)
        
        causal_effect = treated_data[outcome].mean() - reference_data[outcome].mean()
        return causal_effect
    
    def counterfactual(self, evidence: Dict[str, torch.Tensor], 
                      intervention: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise = self._abduct_noise(evidence)
        
        counterfactual_data = {}
        for var in self.graph.topological_order:
            if var in intervention:
                counterfactual_data[var] = intervention[var]
            else:
                parents = self.graph.get_parents(var)
                if parents:
                    parent_inputs = torch.cat([counterfactual_data[parent] for parent in parents], dim=-1)
                    mechanism_input = torch.cat([parent_inputs, noise[var]], dim=-1)
                else:
                    mechanism_input = noise[var]
                    
                counterfactual_data[var] = self.mechanisms[var](mechanism_input)
                
        return counterfactual_data
    
    def _abduct_noise(self, evidence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        noise = {}
        
        for var in self.graph.topological_order:
            if var in evidence:
                parents = self.graph.get_parents(var)
                if parents:
                    parent_inputs = torch.cat([evidence[parent] for parent in parents], dim=-1)
                    deterministic = self.mechanisms[var](torch.cat([parent_inputs, torch.zeros_like(evidence[var])], dim=-1))
                else:
                    deterministic = self.mechanisms[var](torch.zeros_like(evidence[var]))
                    
                noise[var] = evidence[var] - deterministic
            else:
                noise[var] = torch.randn_like(evidence[list(evidence.keys())[0]]) * torch.abs(self.noise_stds[var])
                
        return noise