import torch
import numpy as np
from sklearn.metrics import accuracy_score

class MembershipInferenceAttack:
    def __init__(self, model, attack_type='confidence'):
        self.model = model
        self.attack_type = attack_type
    
    def confidence_attack(self, member_data, nonmember_data):
        self.model.eval()
        
        with torch.no_grad():
            mem_out = torch.softmax(self.model(member_data), dim=1)
            non_out = torch.softmax(self.model(nonmember_data), dim=1)
            
            mem_conf = mem_out.max(dim=1)[0].numpy()
            non_conf = non_out.max(dim=1)[0].numpy()
        
        threshold = np.median(mem_conf)
        
        mem_pred = (mem_conf > threshold).mean()
        non_pred = (non_conf <= threshold).mean()
        
        return (mem_pred + non_pred) * 50
    
    def loss_attack(self, member_data, nonmember_data, member_labels, nonmember_labels):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            mem_out = self.model(member_data)
            non_out = self.model(nonmember_data)
            
            mem_loss = [criterion(mem_out[i:i+1], member_labels[i:i+1]).item() 
                       for i in range(len(member_data))]
            non_loss = [criterion(non_out[i:i+1], nonmember_labels[i:i+1]).item() 
                       for i in range(len(nonmember_data))]
        
        threshold = np.median(mem_loss)
        
        mem_pred = np.array(mem_loss < threshold).mean()
        non_pred = np.array(non_loss >= threshold).mean()
        
        return (mem_pred + non_pred) * 50
    
    def evaluate(self, member_data, nonmember_data, member_labels=None, nonmember_labels=None):
        if self.attack_type == 'confidence':
            return self.confidence_attack(member_data, nonmember_data)
        elif self.attack_type == 'loss':
            return self.loss_attack(member_data, nonmember_data, member_labels, nonmember_labels)
        else:
            raise ValueError("Attack type must be 'confidence' or 'loss'")
