# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import numpy as np
from typing import Dict, List, Optional

class CausalVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_causal_graph(self, graph: nx.DiGraph, title: str = "Causal Graph"):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, ax=self.ax, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=12, font_weight='bold', arrows=True)
        
        self.ax.set_title(title)
        plt.tight_layout()
        return self.fig
    
    def plot_intervention_effect(self, model, treatment: str, outcome: str,
                               treatment_range: List[float], reference_value: float = 0.0,
                               n_samples: int = 1000):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        effects = []
        for treatment_value in treatment_range:
            effect = model.causal_effect(treatment, outcome, treatment_value, reference_value, n_samples)
            effects.append(effect.item())
            
        self.ax.plot(treatment_range, effects, 'o-', linewidth=2, markersize=8)
        self.ax.set_xlabel(f'Treatment ({treatment}) Value')
        self.ax.set_ylabel(f'Causal Effect on {outcome}')
        self.ax.set_title(f'Causal Effect of {treatment} on {outcome}')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig
    
    def plot_counterfactual_distribution(self, model, evidence: Dict[str, torch.Tensor],
                                       intervention: Dict[str, torch.Tensor],
                                       variable: str, n_samples: int = 1000):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        counterfactual_samples = model.counterfactual(evidence, intervention)
        counterfactual_vals = counterfactual_samples[variable].detach().cpu().numpy().flatten()
        
        self.ax.hist(counterfactual_vals, bins=50, alpha=0.7, density=True)
        self.ax.set_xlabel(f'{variable} Value')
        self.ax.set_ylabel('Density')
        self.ax.set_title(f'Counterfactual Distribution of {variable}')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig

class InterventionVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_intervention_comparison(self, true_data: Dict[str, torch.Tensor],
                                   intervened_data: Dict[str, torch.Tensor],
                                   variables: List[str]):
        n_vars = len(variables)
        self.fig, self.axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 5))
        
        if n_vars == 1:
            self.axes = [self.axes]
            
        for i, var in enumerate(variables):
            true_vals = true_data[var].detach().cpu().numpy().flatten()
            intervened_vals = intervened_data[var].detach().cpu().numpy().flatten()
            
            self.axes[i].hist(true_vals, bins=50, alpha=0.7, label='Observational', density=True)
            self.axes[i].hist(intervened_vals, bins=50, alpha=0.7, label='Interventional', density=True)
            self.axes[i].set_xlabel(f'{var} Value')
            self.axes[i].set_ylabel('Density')
            self.axes[i].set_title(f'Distribution of {var}')
            self.axes[i].legend()
            self.axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        return self.fig
    
    def plot_causal_forest(self, model, treatment: str, outcome: str,
                          covariates: List[str], n_points: int = 100):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        treatment_range = torch.linspace(-2, 2, n_points)
        effects = []
        
        for treatment_val in treatment_range:
            effect = model.causal_effect(treatment, outcome, treatment_val.item(), 0.0)
            effects.append(effect.item())
            
        self.ax.plot(treatment_range.numpy(), effects, linewidth=2)
        self.ax.fill_between(treatment_range.numpy(), np.array(effects) - 0.1, 
                           np.array(effects) + 0.1, alpha=0.3)
        self.ax.set_xlabel(f'{treatment} Value')
        self.ax.set_ylabel(f'Causal Effect on {outcome}')
        self.ax.set_title('Causal Effect Function')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig