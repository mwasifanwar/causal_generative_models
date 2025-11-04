# utils/training_utils.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import numpy as np

class CausalTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 device: str = 'cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, 
                   loss_fn: Callable) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            if isinstance(x, dict):
                x = {k: v.to(self.device) for k, v in x.items()}
            else:
                x = x.to(self.device)
                
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward'):
                output = self.model(x)
            else:
                output = self.model(x)
                
            loss = loss_fn(output, x)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: torch.utils.data.DataLoader,
                loss_fn: Callable) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                    
                if isinstance(x, dict):
                    x = {k: v.to(self.device) for k, v in x.items()}
                else:
                    x = x.to(self.device)
                    
                if hasattr(self.model, 'forward'):
                    output = self.model(x)
                else:
                    output = self.model(x)
                    
                loss = loss_fn(output, x)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train(self, train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             loss_fn: Callable, epochs: int,
             early_stopping: Optional[int] = None) -> Dict[str, List[float]]:
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, loss_fn)
            val_loss = self.validate(val_loader, loss_fn)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if early_stopping and patience_counter >= early_stopping:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
        self.model.load_state_dict(torch.load('best_model.pth'))
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}

class InterventionTrainer(CausalTrainer):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str = 'cuda'):
        super().__init__(model, optimizer, device)
        
    def train_epoch_interventional(self, dataloader: torch.utils.data.DataLoader,
                                 loss_fn: Callable, intervention_prob: float = 0.1) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            if isinstance(x, dict):
                x = {k: v.to(self.device) for k, v in x.items()}
            else:
                x = x.to(self.device)
                
            self.optimizer.zero_grad()
            
            interventions = {}
            if isinstance(x, dict) and torch.rand(1) < intervention_prob:
                intervention_var = np.random.choice(list(x.keys()))
                intervention_value = torch.randn_like(x[intervention_var][:1])
                interventions[intervention_var] = intervention_value
                
            if hasattr(self.model, 'sample'):
                if interventions:
                    generated = self.model.sample(x[list(x.keys())[0]].size(0), interventions)
                else:
                    generated = self.model.sample(x[list(x.keys())[0]].size(0))
            else:
                generated = self.model(x)
                
            loss = loss_fn(generated, x)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

class CounterfactualTrainer(CausalTrainer):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str = 'cuda'):
        super().__init__(model, optimizer, device)
        
    def train_epoch_counterfactual(self, dataloader: torch.utils.data.DataLoader,
                                 loss_fn: Callable, counterfactual_prob: float = 0.1) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            if isinstance(x, dict):
                x = {k: v.to(self.device) for k, v in x.items()}
            else:
                x = x.to(self.device)
                
            self.optimizer.zero_grad()
            
            interventions = {}
            if isinstance(x, dict) and torch.rand(1) < counterfactual_prob:
                intervention_var = np.random.choice(list(x.keys()))
                intervention_value = torch.randn_like(x[intervention_var][:1])
                interventions[intervention_var] = intervention_value
                
            if hasattr(self.model, 'counterfactual') and interventions:
                counterfactual = self.model.counterfactual(x, interventions)
                loss = loss_fn(counterfactual, x)
            else:
                if hasattr(self.model, 'forward'):
                    output = self.model(x)
                else:
                    output = self.model(x)
                loss = loss_fn(output, x)
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)