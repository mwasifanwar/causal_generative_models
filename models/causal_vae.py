# models/causal_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from ..core.causal_graphs import CausalGraph

class CausalEncoder(nn.Module):
    def __init__(self, input_dims: Dict[str, int], latent_dims: Dict[str, int], 
                 causal_graph: CausalGraph, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.causal_graph = causal_graph
        
        self.encoders = nn.ModuleDict()
        self.fc_means = nn.ModuleDict()
        self.fc_logvars = nn.ModuleDict()
        
        for var in causal_graph.variables:
            input_dim = input_dims[var]
            latent_dim = latent_dims[var]
            
            encoder_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                encoder_layers.append(nn.ReLU())
                prev_dim = hidden_dim
                
            self.encoders[var] = nn.Sequential(*encoder_layers)
            self.fc_means[var] = nn.Linear(prev_dim, latent_dim)
            self.fc_logvars[var] = nn.Linear(prev_dim, latent_dim)
            
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        means = {}
        logvars = {}
        
        for var in self.causal_graph.topological_order:
            encoded = self.encoders[var](x[var])
            means[var] = self.fc_means[var](encoded)
            logvars[var] = self.fc_logvars[var](encoded)
            
        return means, logvars
    
    def reparameterize(self, means: Dict[str, torch.Tensor], logvars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        latents = {}
        
        for var in means:
            std = torch.exp(0.5 * logvars[var])
            eps = torch.randn_like(std)
            latents[var] = means[var] + eps * std
            
        return latents

class CausalDecoder(nn.Module):
    def __init__(self, latent_dims: Dict[str, int], output_dims: Dict[str, int],
                 causal_graph: CausalGraph, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.causal_graph = causal_graph
        
        self.decoders = nn.ModuleDict()
        
        for var in causal_graph.variables:
            parents = causal_graph.get_parents(var)
            input_dim = latent_dims[var] + sum(latent_dims[parent] for parent in parents)
            output_dim = output_dims[var]
            
            decoder_layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                decoder_layers.append(nn.ReLU())
                prev_dim = hidden_dim
                
            decoder_layers.append(nn.Linear(prev_dim, output_dim))
            self.decoders[var] = nn.Sequential(*decoder_layers)
            
    def forward(self, latents: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reconstructions = {}
        
        for var in self.causal_graph.topological_order:
            parents = self.causal_graph.get_parents(var)
            
            if parents:
                parent_latents = torch.cat([latents[parent] for parent in parents], dim=-1)
                decoder_input = torch.cat([latents[var], parent_latents], dim=-1)
            else:
                decoder_input = latents[var]
                
            reconstructions[var] = self.decoders[var](decoder_input)
            
        return reconstructions

class CausalVAE(nn.Module):
    def __init__(self, input_dims: Dict[str, int], latent_dims: Dict[str, int],
                 causal_graph: CausalGraph, hidden_dims: List[int] = [64, 64]):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.causal_graph = causal_graph
        
        self.encoder = CausalEncoder(input_dims, latent_dims, causal_graph, hidden_dims)
        self.decoder = CausalDecoder(latent_dims, input_dims, causal_graph, hidden_dims)
        
    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        means, logvars = self.encoder(x)
        latents = self.encoder.reparameterize(means, logvars)
        reconstructions = self.decoder(latents)
        
        return reconstructions, means, logvars
    
    def loss_function(self, x: Dict[str, torch.Tensor], reconstructions: Dict[str, torch.Tensor],
                     means: Dict[str, torch.Tensor], logvars: Dict[str, torch.Tensor], 
                     kl_weight: float = 1.0) -> torch.Tensor:
        recon_loss = 0.0
        kl_loss = 0.0
        
        for var in self.causal_graph.variables:
            recon_loss += F.mse_loss(reconstructions[var], x[var], reduction='sum')
            kl_loss += -0.5 * torch.sum(1 + logvars[var] - means[var].pow(2) - logvars[var].exp())
            
        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss
    
    def sample(self, n_samples: int, interventions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        latents = {}
        
        for var in self.causal_graph.topological_order:
            if interventions is not None and var in interventions:
                latents[var] = interventions[var]
            else:
                latent_dim = self.latent_dims[var]
                latents[var] = torch.randn(n_samples, latent_dim)
                
        return self.decoder(latents)
    
    def counterfactual(self, x: Dict[str, torch.Tensor], interventions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        means, logvars = self.encoder(x)
        latents = self.encoder.reparameterize(means, logvars)
        
        for var in interventions:
            latents[var] = interventions[var]
            
        return self.decoder(latents)
    
    def causal_effect(self, treatment: str, outcome: str, 
                     treatment_value: float, reference_value: float,
                     n_samples: int = 1000) -> torch.Tensor:
        treatment_latent = torch.tensor([[treatment_value]], dtype=torch.float32).expand(n_samples, -1)
        reference_latent = torch.tensor([[reference_value]], dtype=torch.float32).expand(n_samples, -1)
        
        treated_samples = self.sample(n_samples, {treatment: treatment_latent})
        reference_samples = self.sample(n_samples, {treatment: reference_latent})
        
        causal_effect = treated_samples[outcome].mean() - reference_samples[outcome].mean()
        return causal_effect