import torch
import torch.nn as nn
from tqdm import tqdm

class Engine:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for batch in tqdm(data_loader, leave=False):
            self.model.zero_grad()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            final_loss += loss.item()        
        return final_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            for batch in tqdm(data_loader, leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                final_loss += loss.item()
            
            return final_loss / len(data_loader)