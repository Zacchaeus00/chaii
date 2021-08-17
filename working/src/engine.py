import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

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
        for batch in tqdm(data_loader, leave=False, miniters=(len(data_loader)//100)):
            self.model.zero_grad()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            final_loss += loss.item()        
        return final_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            for batch in tqdm(data_loader, leave=False, miniters=(len(data_loader)//100)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                final_loss += loss.item()
            
            return final_loss / len(data_loader)

    def predict(self, data_loader):
        with torch.no_grad():
            start_logits = []
            end_logits = []
            self.model.eval()
            for batch in tqdm(data_loader, leave=False, miniters=(len(data_loader)//100)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                start_logits.append(outputs.start_logits)
                end_logits.append(outputs.end_logits)
            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            return [start_logits, end_logits]

    def save(self, path):
        path = Path(path)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

class CustomModelEngine:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for batch in tqdm(data_loader, leave=False, miniters=(len(data_loader)//100)):
            self.model.zero_grad()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs['loss']
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            final_loss += loss.item()        
        return final_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            for batch in tqdm(data_loader, leave=False, miniters=(len(data_loader)//100)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs['loss']
                final_loss += loss.item()
            
            return final_loss / len(data_loader)

    def predict(self, data_loader):
        with torch.no_grad():
            start_logits = []
            end_logits = []
            self.model.eval()
            for batch in tqdm(data_loader, leave=False, miniters=(len(data_loader)//100)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                start_logits.append(outputs['start_logits'])
                end_logits.append(outputs['end_logits'])
            start_logits = torch.cat(start_logits, dim=0)
            end_logits = torch.cat(end_logits, dim=0)
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()
            return [start_logits, end_logits]

    def save(self, path):
        path = Path(path)
        path.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)