# core/causal_processes.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from .causal_graphs import CausalGraph, StructuralCausalModel

class CausalProcess:
    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm
        
    def sample_observational(self, n_samples: int) -> Dict[str, torch.Tensor]:
        return self.scm.sample(n_samples)
    
    def sample_interventional(self, intervention: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
        intervened_scm = self.scm.do_intervention(intervention)
        return intervened_scm.sample(n_samples)
    
    def sample_counterfactual(self, evidence: Dict[str, torch.Tensor], intervention: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.scm.counterfactual(evidence, intervention)
    
    def get_causal_effect(self, treatment: str, outcome: str, treatment_value: float, reference_value: float) -> float:
        n_samples = 1000
        
        treatment_intervention = torch.tensor([[treatment_value]], dtype=torch.float32)
        reference_intervention = torch.tensor([[reference_value]], dtype=torch.float32)
        
        treated_data = self.sample_interventional({treatment: treatment_intervention}, n_samples)
        reference_data = self.sample_interventional({treatment: reference_intervention}, n_samples)
        
        causal_effect = treated_data[outcome].mean() - reference_data[outcome].mean()
        return causal_effect.item()
    
    def get_conditional_causal_effect(self, treatment: str, outcome: str, condition: Dict[str, torch.Tensor], 
                                    treatment_value: float, reference_value: float) -> float:
        n_samples = condition[list(condition.keys())[0]].shape[0]
        
        treatment_intervention = torch.tensor([[treatment_value]], dtype=torch.float32).expand(n_samples, -1)
        reference_intervention = torch.tensor([[reference_value]], dtype=torch.float32).expand(n_samples, -1)
        
        treated_counterfactual = self.sample_counterfactual(condition, {treatment: treatment_intervention})
        reference_counterfactual = self.sample_counterfactual(condition, {treatment: reference_intervention})
        
        causal_effect = (treated_counterfactual[outcome] - reference_counterfactual[outcome]).mean()
        return causal_effect.item()

class CounterfactualProcess:
    def __init__(self, scm: StructuralCausalModel, evidence: Dict[str, torch.Tensor]):
        self.scm = scm
        self.evidence = evidence
        self.noise = self.scm._abduct_noise(evidence)
        
    def sample(self, intervention: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        intervened_scm = self.scm.do_intervention(intervention)
        return intervened_scm.sample(self.evidence[list(self.evidence.keys())[0]].shape[0], self.noise)
    
    def get_counterfactual_distribution(self, intervention: Dict[str, torch.Tensor], variable: str) -> torch.Tensor:
        counterfactual_samples = self.sample(intervention)
        return counterfactual_samples[variable]

class DoCalculus:
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        
    def rule1(self, y: str, z: str, w: str, x: str) -> bool:
        if not self.graph.is_d_separated(y, z, w + [x]):
            return False
            
        intervened_graph = self.graph.intervene({x: 0.0})
        return intervened_graph.is_d_separated(y, z, w)
    
    def rule2(self, y: str, z: str, w: str, x: str) -> bool:
        if not self.graph.is_d_separated(y, z, w + [x]):
            return False
            
        intervened_graph = self.graph.intervene({})
        descendants_x = self.graph.get_descendants(x)
        relevant_nodes = set(w) - set(descendants_x)
        
        return intervened_graph.is_d_separated(y, z, list(relevant_nodes))
    
    def rule3(self, y: str, z: str, w: str, x: str) -> bool:
        if not self.graph.is_d_separated(y, z, w + [x]):
            return False
            
        descendants_x = self.graph.get_descendants(x)
        if any(node in w for node in descendants_x):
            return False
            
        intervened_graph = self.graph.intervene({})
        return intervened_graph.is_d_separated(y, z, w)
    
    def identify_effect(self, treatment: str, outcome: str, covariates: List[str]) -> Optional[str]:
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)
        
        if not backdoor_paths:
            return f"P({outcome}|do({treatment})) = P({outcome}|{treatment})"
        
        sufficient_sets = self._find_sufficient_sets(treatment, outcome, covariates)
        
        if sufficient_sets:
            best_set = min(sufficient_sets, key=len)
            return f"P({outcome}|do({treatment})) = Σ_{{{','.join(best_set)}}} P({outcome}|{treatment},{','.join(best_set)})P({','.join(best_set)})"
        
        frontdoor_sets = self._find_frontdoor_sets(treatment, outcome, covariates)
        
        if frontdoor_sets:
            best_set = min(frontdoor_sets, key=len)
            return f"P({outcome}|do({treatment})) = Σ_{{{','.join(best_set)}}} P({outcome}|do({','.join(best_set)}))P({','.join(best_set)}|do({treatment}))"
            
        return None
    
    def _find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        all_paths = []
        
        for path in nx.all_simple_paths(self.graph.graph, treatment, outcome):
            if len(path) > 2:
                all_paths.append(path)
                
        return all_paths
    
    def _find_sufficient_sets(self, treatment: str, outcome: str, covariates: List[str]) -> List[List[str]]:
        sufficient_sets = []
        
        for k in range(len(covariates) + 1):
            for candidate_set in itertools.combinations(covariates, k):
                candidate_list = list(candidate_set)
                
                blocks_all_backdoor = True
                for path in self._find_backdoor_paths(treatment, outcome):
                    if not self._blocks_path(path, candidate_list):
                        blocks_all_backdoor = False
                        break
                        
                if blocks_all_backdoor and not self._contains_descendants(treatment, candidate_list):
                    sufficient_sets.append(candidate_list)
                    
        return sufficient_sets
    
    def _find_frontdoor_sets(self, treatment: str, outcome: str, covariates: List[str]) -> List[List[str]]:
        frontdoor_sets = []
        
        mediator_candidates = [var for var in covariates if var != treatment and var != outcome]
        
        for k in range(1, len(mediator_candidates) + 1):
            for candidate_set in itertools.combinations(mediator_candidates, k):
                candidate_list = list(candidate_set)
                
                if (self._blocks_all_directed_paths(treatment, outcome, candidate_list) and
                    self._no_backdoor_treatment_mediator(candidate_list, treatment) and
                    self._no_backdoor_mediator_outcome(candidate_list, outcome)):
                    frontdoor_sets.append(candidate_list)
                    
        return frontdoor_sets
    
    def _blocks_path(self, path: List[str], condition_set: List[str]) -> bool:
        for i in range(1, len(path) - 1):
            node = path[i]
            if node in condition_set:
                return True
                
            parents = self.graph.get_parents(node)
            children = self.graph.get_children(node)
            
            if (set(parents) & set(path)) and (set(children) & set(path)):
                if node not in condition_set:
                    return False
                    
        return True
    
    def _contains_descendants(self, treatment: str, condition_set: List[str]) -> bool:
        descendants = self.graph.get_descendants(treatment)
        return any(var in descendants for var in condition_set)
    
    def _blocks_all_directed_paths(self, treatment: str, outcome: str, mediators: List[str]) -> bool:
        for path in nx.all_simple_paths(self.graph.graph, treatment, outcome):
            if not any(mediator in path for mediator in mediators):
                return False
        return True
    
    def _no_backdoor_treatment_mediator(self, mediators: List[str], treatment: str) -> bool:
        for mediator in mediators:
            backdoor_paths = self._find_backdoor_paths(treatment, mediator)
            if backdoor_paths:
                return False
        return True
    
    def _no_backdoor_mediator_outcome(self, mediators: List[str], outcome: str) -> bool:
        for mediator in mediators:
            descendants = self.graph.get_descendants(mediator)
            if outcome in descendants:
                backdoor_paths = self._find_backdoor_paths(mediator, outcome)
                if backdoor_paths:
                    return False
        return True