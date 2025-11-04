# examples/synthetic_experiments.py
import torch
import torch.nn as nn
import numpy as np
from ..core.causal_graphs import CausalGraph, StructuralCausalModel
from ..core.causal_mechanisms import NeuralCausalModel
from ..models.causal_vae import CausalVAE
from ..utils.training_utils import CausalTrainer
from ..utils.evaluation_metrics import CausalMetrics

def linear_scm_experiment():
    print("Linear Structural Causal Model Experiment")
    
    variables = ['X', 'Y', 'Z']
    edges = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
    
    graph = CausalGraph(variables, edges)
    
    mechanisms = {
        'X': nn.Linear(1, 1),
        'Y': nn.Linear(2, 1),
        'Z': nn.Linear(3, 1)
    }
    
    mechanisms['X'].weight.data = torch.tensor([[0.5]])
    mechanisms['X'].bias.data = torch.tensor([0.0])
    
    mechanisms['Y'].weight.data = torch.tensor([[0.7, 0.3]])
    mechanisms['Y'].bias.data = torch.tensor([0.0])
    
    mechanisms['Z'].weight.data = torch.tensor([[0.4, 0.5, 0.1]])
    mechanisms['Z'].bias.data = torch.tensor([0.0])
    
    scm = StructuralCausalModel(graph, mechanisms)
    
    for var in variables:
        scm.set_noise_distribution(var, torch.distributions.Normal(0, 0.1))
        
    observational_data = scm.sample(1000)
    
    print("Observational data means:")
    for var in variables:
        print(f"{var}: {observational_data[var].mean().item():.3f}")
        
    intervention = {'X': torch.tensor([[2.0]])}
    interventional_data = scm.sample_interventional(intervention, 1000)
    
    print("\nInterventional data means (do(X=2)):")
    for var in variables:
        print(f"{var}: {interventional_data[var].mean().item():.3f}")
        
    evidence = {var: observational_data[var][:10] for var in variables}
    counterfactual_data = scm.counterfactual(evidence, intervention)
    
    print("\nCounterfactual example (what if X=2 for first 10 samples):")
    for var in variables:
        print(f"{var} - Original: {evidence[var][0].item():.3f}, Counterfactual: {counterfactual_data[var][0].item():.3f}")
        
    return scm, observational_data, interventional_data, counterfactual_data

def nonlin_scm_experiment():
    print("\nNonlinear Structural Causal Model Experiment")
    
    variables = ['A', 'B', 'C']
    edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    
    graph = CausalGraph(variables, edges)
    
    mechanisms = {
        'A': nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)),
        'B': nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)),
        'C': nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    }
    
    neural_scm = NeuralCausalModel(variables, edges, {'A': 1, 'B': 1, 'C': 1})
    
    n_samples = 5000
    observational_data = neural_scm.sample(n_samples)
    
    print("Neural SCM Observational data means:")
    for var in variables:
        print(f"{var}: {observational_data[var].mean().item():.3f}")
        
    intervention = {'A': torch.tensor([[1.5]])}
    interventional_data = neural_scm.sample(n_samples, intervention)
    
    print("\nNeural SCM Interventional data means (do(A=1.5)):")
    for var in variables:
        print(f"{var}: {interventional_data[var].mean().item():.3f}")
        
    causal_effect = neural_scm.causal_effect('A', 'C', 1.5, 0.0)
    print(f"\nCausal effect of A on C: {causal_effect.item():.3f}")
    
    input_dims = {'A': 1, 'B': 1, 'C': 1}
    latent_dims = {'A': 2, 'B': 2, 'C': 2}
    
    causal_vae = CausalVAE(input_dims, latent_dims, graph)
    optimizer = torch.optim.Adam(causal_vae.parameters(), lr=0.001)
    
    def vae_loss(reconstructions, x, means, logvars):
        return causal_vae.loss_function(x, reconstructions, means, logvars)
    
    dataset = torch.utils.data.TensorDataset(
        torch.cat([observational_data[var] for var in variables], dim=-1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    trainer = CausalTrainer(causal_vae, optimizer)
    losses = trainer.train(dataloader, dataloader, vae_loss, epochs=100)
    
    print("\nCausal VAE training completed")
    print(f"Final training loss: {losses['train_losses'][-1]:.4f}")
    
    return neural_scm, causal_vae, observational_data, interventional_data