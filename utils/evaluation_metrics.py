# utils/evaluation_metrics.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import wasserstein_distance

class CausalMetrics:
    def __init__(self, true_scm, estimated_scm):
        self.true_scm = true_scm
        self.estimated_scm = estimated_scm
        
    def interventional_distance(self, intervention: Dict[str, float], 
                               n_samples: int = 1000) -> float:
        true_intervened = self.true_scm.sample_interventional(intervention, n_samples)
        estimated_intervened = self.estimated_scm.sample_interventional(intervention, n_samples)
        
        distance = 0.0
        for var in true_intervened:
            true_samples = true_intervened[var].detach().cpu().numpy().flatten()
            estimated_samples = estimated_intervened[var].detach().cpu().numpy().flatten()
            distance += wasserstein_distance(true_samples, estimated_samples)
            
        return distance / len(true_intervened)
    
    def counterfactual_accuracy(self, evidence: Dict[str, torch.Tensor],
                               intervention: Dict[str, torch.Tensor],
                               threshold: float = 0.1) -> float:
        true_counterfactual = self.true_scm.sample_counterfactual(evidence, intervention)
        estimated_counterfactual = self.estimated_scm.sample_counterfactual(evidence, intervention)
        
        correct = 0
        total = 0
        
        for var in true_counterfactual:
            true_vals = true_counterfactual[var]
            estimated_vals = estimated_counterfactual[var]
            
            differences = torch.abs(true_vals - estimated_vals)
            correct += torch.sum(differences < threshold).item()
            total += true_vals.numel()
            
        return correct / total if total > 0 else 0.0
    
    def causal_effect_error(self, treatment: str, outcome: str,
                           treatment_values: List[float], 
                           reference_value: float = 0.0) -> float:
        errors = []
        
        for treatment_value in treatment_values:
            true_effect = self.true_scm.get_causal_effect(treatment, outcome, 
                                                         treatment_value, reference_value)
            estimated_effect = self.estimated_scm.get_causal_effect(treatment, outcome,
                                                                   treatment_value, reference_value)
            errors.append(abs(true_effect - estimated_effect))
            
        return np.mean(errors)
    
    def structural_accuracy(self) -> float:
        true_edges = set(self.true_scm.graph.edges)
        estimated_edges = set(self.estimated_scm.graph.edges)
        
        intersection = true_edges & estimated_edges
        union = true_edges | estimated_edges
        
        return len(intersection) / len(union) if union else 1.0

class InterventionMetrics:
    def __init__(self, model, true_data_generator):
        self.model = model
        self.true_data_generator = true_data_generator
        
    def intervention_consistency(self, interventions: List[Dict[str, float]],
                               n_samples: int = 1000) -> float:
        consistencies = []
        
        for intervention in interventions:
            true_samples = self.true_data_generator.sample_interventional(intervention, n_samples)
            model_samples = self.model.sample(n_samples, intervention)
            
            consistency = 0.0
            for var in true_samples:
                if var in model_samples:
                    true_mean = true_samples[var].mean().item()
                    model_mean = model_samples[var].mean().item()
                    consistency += 1 - abs(true_mean - model_mean) / (abs(true_mean) + 1e-8)
                    
            consistencies.append(consistency / len(true_samples))
            
        return np.mean(consistencies)
    
    def intervention_fidelity(self, base_data: Dict[str, torch.Tensor],
                            interventions: List[Dict[str, float]]) -> float:
        fidelities = []
        
        for intervention in interventions:
            intervened_vars = set(intervention.keys())
            
            model_counterfactual = self.model.counterfactual(base_data, intervention)
            
            fidelity = 0.0
            for var in base_data:
                if var not in intervened_vars:
                    original_corr = torch.corrcoef(torch.stack([base_data[var].flatten(), 
                                                              base_data[var].flatten()]))[0, 1]
                    counterfactual_corr = torch.corrcoef(torch.stack([base_data[var].flatten(),
                                                                    model_counterfactual[var].flatten()]))[0, 1]
                    fidelity += abs(original_corr - counterfactual_corr).item()
                    
            fidelities.append(1 - fidelity / (len(base_data) - len(intervened_vars)))
            
        return np.mean(fidelities)

class CounterfactualMetrics:
    def __init__(self, model, true_scm):
        self.model = model
        self.true_scm = true_scm
        
    def counterfactual_consistency(self, evidence_list: List[Dict[str, torch.Tensor]],
                                 intervention_list: List[Dict[str, torch.Tensor]]) -> float:
        consistencies = []
        
        for evidence, intervention in zip(evidence_list, intervention_list):
            true_counterfactual = self.true_scm.sample_counterfactual(evidence, intervention)
            
            if hasattr(self.model, 'counterfactual'):
                model_counterfactual = self.model.counterfactual(evidence, intervention)
            else:
                model_counterfactual = self.model.sample(evidence[list(evidence.keys())[0]].size(0), intervention)
                
            consistency = 0.0
            for var in true_counterfactual:
                if var in model_counterfactual:
                    true_vals = true_counterfactual[var]
                    model_vals = model_counterfactual[var]
                    correlation = torch.corrcoef(torch.stack([true_vals.flatten(), model_vals.flatten()]))[0, 1]
                    consistency += correlation.item() if not torch.isnan(correlation) else 0.0
                    
            consistencies.append(consistency / len(true_counterfactual))
            
        return np.mean(consistencies)
    
    def factual_accuracy(self, test_data: Dict[str, torch.Tensor]) -> float:
        if hasattr(self.model, 'log_prob'):
            log_prob = self.model.log_prob(test_data)
            return torch.exp(log_prob).mean().item()
        else:
            return 0.0