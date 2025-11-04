# models/causal_flows.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from ..core.causal_graphs import CausalGraph

class CausalCouplingLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64], mask: torch.Tensor = None):
        super().__init__()
        self.input_dim = input_dim
        
        if mask is None:
            mask = torch.ones(input_dim)
            mask[::2] = 0
        self.register_buffer('mask', mask)
        
        self.scale_net = self._build_network(input_dim, hidden_dims, input_dim)
        self.translate_net = self._build_network(input_dim, hidden_dims, input_dim)
        
    def _build_network(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        
        if not reverse:
            log_scale = torch.tanh(self.scale_net(x_masked)) * (1 - self.mask)
            translate = self.translate_net(x_masked) * (1 - self.mask)
            
            y = x_masked + (1 - self.mask) * (x * torch.exp(log_scale) + translate)
            log_det = torch.sum(log_scale, dim=-1)
        else:
            y_masked = x * self.mask
            
            log_scale = torch.tanh(self.scale_net(y_masked)) * (1 - self.mask)
            translate = self.translate_net(y_masked) * (1 - self.mask)
            
            y = y_masked + (1 - self.mask) * ((x - translate) * torch.exp(-log_scale))
            log_det = -torch.sum(log_scale, dim=-1)
            
        return y, log_det

class CausalFlow(nn.Module):
    def __init__(self, variable_dims: Dict[str, int], causal_graph: CausalGraph,
                 num_layers: int = 4, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.variable_dims = variable_dims
        self.causal_graph = causal_graph
        self.num_layers = num_layers
        
        self.flows = nn.ModuleDict()
        self.base_distributions = nn.ModuleDict()
        
        total_dim = sum(variable_dims.values())
        
        for var in causal_graph.variables:
            input_dim = variable_dims[var]
            
            var_flows = []
            for i in range(num_layers):
                mask = torch.ones(total_dim)
                start_idx = 0
                for v in causal_graph.variables:
                    if v == var:
                        mask[start_idx:start_idx + variable_dims[v]] = 0 if i % 2 == 0 else 1
                    else:
                        mask[start_idx:start_idx + variable_dims[v]] = 1 if i % 2 == 0 else 0
                    start_idx += variable_dims[v]
                    
                var_flows.append(CausalCouplingLayer(total_dim, hidden_dims, mask))
                
            self.flows[var] = nn.ModuleList(var_flows)
            self.base_distributions[var] = torch.distributions.Normal(0, 1)
            
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = x[list(x.keys())[0]].size(0)
        total_dim = sum(self.variable_dims.values())
        
        x_combined = torch.cat([x[var] for var in self.causal_graph.variables], dim=-1)
        z_combined = x_combined.clone()
        log_det = torch.zeros(batch_size)
        
        for var in self.causal_graph.topological_order:
            for flow in self.flows[var]:
                z_combined, ld = flow(z_combined)
                log_det += ld
                
        z_dict = {}
        start_idx = 0
        for var in self.causal_graph.variables:
            end_idx = start_idx + self.variable_dims[var]
            z_dict[var] = z_combined[:, start_idx:end_idx]
            start_idx = end_idx
            
        return z_dict, log_det
    
    def inverse(self, z: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = z[list(z.keys())[0]].size(0)
        
        z_combined = torch.cat([z[var] for var in self.causal_graph.variables], dim=-1)
        x_combined = z_combined.clone()
        log_det = torch.zeros(batch_size)
        
        for var in reversed(self.causal_graph.topological_order):
            for flow in reversed(self.flows[var]):
                x_combined, ld = flow(x_combined, reverse=True)
                log_det += ld
                
        x_dict = {}
        start_idx = 0
        for var in self.causal_graph.variables:
            end_idx = start_idx + self.variable_dims[var]
            x_dict[var] = x_combined[:, start_idx:end_idx]
            start_idx = end_idx
            
        return x_dict, log_det
    
    def log_prob(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        z, log_det = self.forward(x)
        
        log_prob = log_det
        for var in self.causal_graph.variables:
            log_prob += self.base_distributions[var].log_prob(z[var]).sum(dim=-1)
            
        return log_prob.mean()
    
    def sample(self, n_samples: int, interventions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        z = {}
        for var in self.causal_graph.variables:
            if interventions is not None and var in interventions:
                z[var] = interventions[var]
            else:
                z[var] = self.base_distributions[var].sample((n_samples, self.variable_dims[var]))
                
        x, _ = self.inverse(z)
        return x

class CausalNormalizingFlow(nn.Module):
    def __init__(self, variable_dims: Dict[str, int], causal_graph: CausalGraph,
                 num_flows: int = 6, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.variable_dims = variable_dims
        self.causal_graph = causal_graph
        self.num_flows = num_flows
        
        self.flow = CausalFlow(variable_dims, causal_graph, num_flows, hidden_dims)
        
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.flow.log_prob(x)
    
    def sample(self, n_samples: int, interventions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        return self.flow.sample(n_samples, interventions)
    
    def counterfactual(self, x: Dict[str, torch.Tensor], interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z, _ = self.flow.forward(x)
        
        for var in interventions:
            z[var] = interventions[var]
            
        counterfactual_x, _ = self.flow.inverse(z)
        return counterfactual_x
    
    def causal_effect(self, treatment: str, outcome: str,
                     treatment_value: float, reference_value: float,
                     n_samples: int = 1000) -> torch.Tensor:
        treatment_tensor = torch.tensor([[treatment_value]], dtype=torch.float32).expand(n_samples, -1)
        reference_tensor = torch.tensor([[reference_value]], dtype=torch.float32).expand(n_samples, -1)
        
        treated_samples = self.sample(n_samples, {treatment: treatment_tensor})
        reference_samples = self.sample(n_samples, {treatment: reference_tensor})
        
        causal_effect = treated_samples[outcome].mean() - reference_samples[outcome].mean()
        return causal_effect