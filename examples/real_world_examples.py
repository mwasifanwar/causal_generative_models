# examples/real_world_examples.py
import torch
import torch.nn as nn
import numpy as np
from ..core.causal_graphs import CausalGraph, StructuralCausalModel
from ..models.causal_gan import CausalGAN
from ..models.causal_flows import CausalNormalizingFlow
from ..utils.training_utils import InterventionTrainer, CounterfactualTrainer

def causal_generation_example():
    print("Causal Generation Example")
    
    variables = ['age', 'income', 'spending']
    edges = [('age', 'income'), ('age', 'spending'), ('income', 'spending')]
    
    graph = CausalGraph(variables, edges)
    
    noise_dims = {'age': 2, 'income': 3, 'spending': 2}
    data_dims = {'age': 1, 'income': 1, 'spending': 1}
    
    causal_gan = CausalGAN(noise_dims, data_dims, graph)
    
    n_samples = 1000
    real_data = {
        'age': torch.randn(n_samples, 1) * 10 + 30,
        'income': torch.randn(n_samples, 1) * 5000 + 30000,
        'spending': torch.randn(n_samples, 1) * 1000 + 5000
    }
    
    real_data['income'] = real_data['income'] + real_data['age'] * 100
    real_data['spending'] = real_data['spending'] + real_data['age'] * 50 + real_data['income'] * 0.1
    
    print("Real data statistics:")
    for var in variables:
        print(f"{var}: mean={real_data[var].mean().item():.1f}, std={real_data[var].std().item():.1f}")
        
    for epoch in range(100):
        d_loss, g_loss = causal_gan.train_step(real_data)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
            
    generated_data = causal_gan.sample(1000)
    
    print("\nGenerated data statistics:")
    for var in variables:
        print(f"{var}: mean={generated_data[var].mean().item():.1f}, std={generated_data[var].std().item():.1f}")
        
    intervention = {'age': torch.tensor([[40.0]])}
    intervened_data = causal_gan.sample(1000, intervention)
    
    print("\nIntervened data (age=40) statistics:")
    for var in variables:
        print(f"{var}: mean={intervened_data[var].mean().item():.1f}")
        
    return causal_gan, real_data, generated_data, intervened_data

def intervention_study():
    print("\nIntervention Study Example")
    
    variables = ['treatment', 'confounder', 'outcome']
    edges = [('confounder', 'treatment'), ('confounder', 'outcome'), ('treatment', 'outcome')]
    
    graph = CausalGraph(variables, edges)
    
    variable_dims = {'treatment': 1, 'confounder': 1, 'outcome': 1}
    
    causal_flow = CausalNormalizingFlow(variable_dims, graph)
    optimizer = torch.optim.Adam(causal_flow.parameters(), lr=0.001)
    
    n_samples = 2000
    real_data = {
        'confounder': torch.randn(n_samples, 1),
        'treatment': torch.randn(n_samples, 1) + 0.5 * torch.randn(n_samples, 1),
        'outcome': torch.randn(n_samples, 1) + 0.3 * real_data['confounder'] + 0.7 * real_data['treatment']
    }
    
    dataset = torch.utils.data.TensorDataset(
        torch.cat([real_data[var] for var in variables], dim=-1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    def flow_loss(output, x):
        return -output
    
    trainer = CounterfactualTrainer(causal_flow, optimizer)
    losses = trainer.train(dataloader, dataloader, flow_loss, epochs=50)
    
    print("Causal Flow training completed")
    
    observational_samples = causal_flow.sample(1000)
    interventional_samples = causal_flow.sample(1000, {'treatment': torch.tensor([[1.0]])})
    
    print("\nObservational vs Interventional means:")
    for var in variables:
        obs_mean = observational_samples[var].mean().item()
        int_mean = interventional_samples[var].mean().item()
        print(f"{var}: Observational={obs_mean:.3f}, Interventional={int_mean:.3f}")
        
    ate = causal_flow.causal_effect('treatment', 'outcome', 1.0, 0.0)
    print(f"\nAverage Treatment Effect: {ate.item():.3f}")
    
    evidence = {var: real_data[var][:10] for var in variables}
    counterfactual = causal_flow.counterfactual(evidence, {'treatment': torch.tensor([[2.0]])})
    
    print("\nCounterfactual example:")
    for var in variables:
        original = evidence[var][0].item()
        counter = counterfactual[var][0].item()
        print(f"{var}: Original={original:.3f}, Counterfactual={counter:.3f}")
        
    return causal_flow, real_data, observational_samples, interventional_samples