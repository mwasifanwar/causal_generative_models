# models/causal_gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from ..core.causal_graphs import CausalGraph

class CausalGenerator(nn.Module):
    def __init__(self, noise_dims: Dict[str, int], output_dims: Dict[str, int],
                 causal_graph: CausalGraph, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.noise_dims = noise_dims
        self.output_dims = output_dims
        self.causal_graph = causal_graph
        
        self.generators = nn.ModuleDict()
        
        for var in causal_graph.variables:
            parents = causal_graph.get_parents(var)
            input_dim = noise_dims[var] + sum(output_dims[parent] for parent in parents)
            output_dim = output_dims[var]
            
            generator_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                generator_layers.append(nn.Linear(prev_dim, hidden_dim))
                generator_layers.append(nn.ReLU())
                prev_dim = hidden_dim
                
            generator_layers.append(nn.Linear(prev_dim, output_dim))
            generator_layers.append(nn.Tanh())
            self.generators[var] = nn.Sequential(*generator_layers)
            
    def forward(self, noise: Optional[Dict[str, torch.Tensor]] = None, 
                interventions: Optional[Dict[str, torch.Tensor]] = None,
                n_samples: int = 1) -> Dict[str, torch.Tensor]:
        if noise is None:
            noise = {}
            for var in self.causal_graph.variables:
                noise[var] = torch.randn(n_samples, self.noise_dims[var])
                
        samples = {}
        
        for var in self.causal_graph.topological_order:
            if interventions is not None and var in interventions:
                samples[var] = interventions[var]
            else:
                parents = self.causal_graph.get_parents(var)
                
                if parents:
                    parent_inputs = torch.cat([samples[parent] for parent in parents], dim=-1)
                    generator_input = torch.cat([noise[var], parent_inputs], dim=-1)
                else:
                    generator_input = noise[var]
                    
                samples[var] = self.generators[var](generator_input)
                
        return samples

class CausalDiscriminator(nn.Module):
    def __init__(self, input_dims: Dict[str, int], causal_graph: CausalGraph,
                 hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.input_dims = input_dims
        self.causal_graph = causal_graph
        
        self.discriminators = nn.ModuleDict()
        
        for var in causal_graph.variables:
            parents = causal_graph.get_parents(var)
            input_dim = input_dims[var] + sum(input_dims[parent] for parent in parents)
            
            discriminator_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                discriminator_layers.append(nn.Linear(prev_dim, hidden_dim))
                discriminator_layers.append(nn.LeakyReLU(0.2))
                prev_dim = hidden_dim
                
            discriminator_layers.append(nn.Linear(prev_dim, 1))
            discriminator_layers.append(nn.Sigmoid())
            self.discriminators[var] = nn.Sequential(*discriminator_layers)
            
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scores = {}
        
        for var in self.causal_graph.topological_order:
            parents = self.causal_graph.get_parents(var)
            
            if parents:
                parent_inputs = torch.cat([x[parent] for parent in parents], dim=-1)
                discriminator_input = torch.cat([x[var], parent_inputs], dim=-1)
            else:
                discriminator_input = x[var]
                
            scores[var] = self.discriminators[var](discriminator_input)
            
        return scores

class CausalGAN:
    def __init__(self, noise_dims: Dict[str, int], data_dims: Dict[str, int],
                 causal_graph: CausalGraph, hidden_dims: List[int] = [64, 64],
                 lr: float = 0.0002, betas: Tuple[float, float] = (0.5, 0.999)):
        self.causal_graph = causal_graph
        self.noise_dims = noise_dims
        self.data_dims = data_dims
        
        self.generator = CausalGenerator(noise_dims, data_dims, causal_graph, hidden_dims)
        self.discriminator = CausalDiscriminator(data_dims, causal_graph, hidden_dims)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        
        self.criterion = nn.BCELoss()
        
    def train_step(self, real_data: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        batch_size = real_data[list(real_data.keys())[0]].size(0)
        
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        self.optimizer_D.zero_grad()
        
        real_scores = self.discriminator(real_data)
        d_loss_real = 0.0
        for var in real_scores:
            d_loss_real += self.criterion(real_scores[var], real_labels)
            
        fake_data = self.generator(n_samples=batch_size)
        fake_scores = self.discriminator(fake_data)
        d_loss_fake = 0.0
        for var in fake_scores:
            d_loss_fake += self.criterion(fake_scores[var], fake_labels)
            
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_D.step()
        
        self.optimizer_G.zero_grad()
        
        fake_data = self.generator(n_samples=batch_size)
        fake_scores = self.discriminator(fake_data)
        
        g_loss = 0.0
        for var in fake_scores:
            g_loss += self.criterion(fake_scores[var], real_labels)
            
        g_loss.backward()
        self.optimizer_G.step()
        
        return d_loss.item(), g_loss.item()
    
    def sample(self, n_samples: int, interventions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        return self.generator(n_samples=n_samples, interventions=interventions)
    
    def counterfactual(self, real_data: Dict[str, torch.Tensor], 
                      interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = real_data[list(real_data.keys())[0]].size(0)
        
        noise = {}
        for var in self.causal_graph.variables:
            noise[var] = torch.randn(batch_size, self.noise_dims[var])
            
        counterfactual_data = {}
        
        for var in self.causal_graph.topological_order:
            if var in interventions:
                counterfactual_data[var] = interventions[var]
            else:
                parents = self.causal_graph.get_parents(var)
                
                if parents:
                    parent_inputs = torch.cat([counterfactual_data[parent] for parent in parents], dim=-1)
                    generator_input = torch.cat([noise[var], parent_inputs], dim=-1)
                else:
                    generator_input = noise[var]
                    
                counterfactual_data[var] = self.generator.generators[var](generator_input)
                
        return counterfactual_data
    
    def causal_effect(self, treatment: str, outcome: str,
                     treatment_value: float, reference_value: float,
                     n_samples: int = 1000) -> torch.Tensor:
        treatment_tensor = torch.tensor([[treatment_value]], dtype=torch.float32).expand(n_samples, -1)
        reference_tensor = torch.tensor([[reference_value]], dtype=torch.float32).expand(n_samples, -1)
        
        treated_samples = self.sample(n_samples, {treatment: treatment_tensor})
        reference_samples = self.sample(n_samples, {treatment: reference_tensor})
        
        causal_effect = treated_samples[outcome].mean() - reference_samples[outcome].mean()
        return causal_effect