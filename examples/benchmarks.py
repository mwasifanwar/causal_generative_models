# examples/benchmarks.py
import torch
import numpy as np
import time
from ..core.causal_graphs import CausalGraph, StructuralCausalModel
from ..models.causal_vae import CausalVAE
from ..models.causal_gan import CausalGAN
from ..models.causal_flows import CausalNormalizingFlow
from ..utils.evaluation_metrics import CausalMetrics, InterventionMetrics

def causal_discovery_benchmark():
    print("Causal Discovery Benchmark")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    variables = ['X1', 'X2', 'X3', 'X4']
    true_edges = [('X1', 'X2'), ('X1', 'X3'), ('X2', 'X4'), ('X3', 'X4')]
    
    true_graph = CausalGraph(variables, true_edges)
    
    mechanisms = {}
    for var in variables:
        parents = true_graph.get_parents(var)
        input_dim = len(parents) + 1
        mechanism = torch.nn.Linear(input_dim, 1)
        mechanisms[var] = mechanism
        
    true_scm = StructuralCausalModel(true_graph, mechanisms)
    
    n_samples = 5000
    true_data = true_scm.sample(n_samples)
    
    models = {}
    training_times = {}
    metrics = {}
    
    input_dims = {var: 1 for var in variables}
    latent_dims = {var: 2 for var in variables}
    
    estimated_graph = CausalGraph(variables, true_edges)
    
    print("Training Causal VAE...")
    start_time = time.time()
    causal_vae = CausalVAE(input_dims, latent_dims, estimated_graph)
    optimizer_vae = torch.optim.Adam(causal_vae.parameters(), lr=0.001)
    
    dataset_vae = torch.utils.data.TensorDataset(
        torch.cat([true_data[var] for var in variables], dim=-1)
    )
    dataloader_vae = torch.utils.data.DataLoader(dataset_vae, batch_size=32, shuffle=True)
    
    for epoch in range(50):
        for batch in dataloader_vae:
            optimizer_vae.zero_grad()
            reconstructions, means, logvars = causal_vae(batch[0])
            loss = causal_vae.loss_function(batch[0], reconstructions, means, logvars)
            loss.backward()
            optimizer_vae.step()
            
    training_times['CausalVAE'] = time.time() - start_time
    models['CausalVAE'] = causal_vae
    
    print("Training Causal GAN...")
    start_time = time.time()
    noise_dims = {var: 2 for var in variables}
    data_dims = {var: 1 for var in variables}
    
    causal_gan = CausalGAN(noise_dims, data_dims, estimated_graph)
    
    for epoch in range(100):
        causal_gan.train_step(true_data)
        
    training_times['CausalGAN'] = time.time() - start_time
    models['CausalGAN'] = causal_gan
    
    print("Training Causal Flow...")
    start_time = time.time()
    variable_dims = {var: 1 for var in variables}
    causal_flow = CausalNormalizingFlow(variable_dims, estimated_graph)
    optimizer_flow = torch.optim.Adam(causal_flow.parameters(), lr=0.001)
    
    dataset_flow = torch.utils.data.TensorDataset(
        torch.cat([true_data[var] for var in variables], dim=-1)
    )
    dataloader_flow = torch.utils.data.DataLoader(dataset_flow, batch_size=32, shuffle=True)
    
    for epoch in range(50):
        for batch in dataloader_flow:
            optimizer_flow.zero_grad()
            log_prob = causal_flow(batch[0])
            loss = -log_prob
            loss.backward()
            optimizer_flow.step()
            
    training_times['CausalFlow'] = time.time() - start_time
    models['CausalFlow'] = causal_flow
    
    print("\nBenchmark Results:")
    print("Model\t\tTraining Time\tInterventional Error\tCounterfactual Acc")
    print("-" * 70)
    
    for model_name, model in models.items():
        estimated_scm = StructuralCausalModel(estimated_graph, {})
        metrics_calculator = CausalMetrics(true_scm, estimated_scm)
        
        intervention_error = metrics_calculator.interventional_distance({'X1': 1.0})
        
        evidence = {var: true_data[var][:10] for var in variables}
        counterfactual_acc = metrics_calculator.counterfactual_accuracy(
            evidence, {'X1': torch.tensor([[2.0]])}
        )
        
        print(f"{model_name}\t{training_times[model_name]:.2f}s\t\t{intervention_error:.4f}\t\t\t{counterfactual_acc:.4f}")
        
    return models, training_times

def intervention_benchmark():
    print("\nIntervention Benchmark")
    
    variables = ['A', 'B', 'C', 'D']
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
    
    graph = CausalGraph(variables, edges)
    
    mechanisms = {}
    for var in variables:
        parents = graph.get_parents(var)
        input_dim = len(parents) + 1
        mechanism = torch.nn.Linear(input_dim, 1)
        mechanisms[var] = mechanism
        
    scm = StructuralCausalModel(graph, mechanisms)
    
    n_samples = 1000
    base_data = scm.sample(n_samples)
    
    interventions = [
        {'A': torch.tensor([[1.0]])},
        {'B': torch.tensor([[2.0]])},
        {'C': torch.tensor([[1.5]])},
        {'A': torch.tensor([[1.0]]), 'B': torch.tensor([[2.0]])}
    ]
    
    models = {}
    
    input_dims = {var: 1 for var in variables}
    latent_dims = {var: 2 for var in variables}
    
    causal_vae = CausalVAE(input_dims, latent_dims, graph)
    models['CausalVAE'] = causal_vae
    
    noise_dims = {var: 2 for var in variables}
    data_dims = {var: 1 for var in variables}
    causal_gan = CausalGAN(noise_dims, data_dims, graph)
    models['CausalGAN'] = causal_gan
    
    variable_dims = {var: 1 for var in variables}
    causal_flow = CausalNormalizingFlow(variable_dims, graph)
    models['CausalFlow'] = causal_flow
    
    print("Intervention Consistency Scores:")
    print("Model\t\tSingle Int 1\tSingle Int 2\tDouble Int")
    print("-" * 60)
    
    for model_name, model in models.items():
        metrics_calculator = InterventionMetrics(model, scm)
        
        single_int1 = metrics_calculator.intervention_consistency([interventions[0]])
        single_int2 = metrics_calculator.intervention_consistency([interventions[1]])
        double_int = metrics_calculator.intervention_consistency([interventions[3]])
        
        print(f"{model_name}\t{single_int1:.4f}\t\t{single_int2:.4f}\t\t{double_int:.4f}")
        
    return models, interventions