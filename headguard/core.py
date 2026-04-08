import torch
import numpy as np
from scipy import signal

class HeadGuard:
    def __init__(self, epsilon=4.0, delta=1e-5, sampling_rate=64, clip_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.fs = sampling_rate
        self.C = clip_norm
        self.weights = None
    
    def compute_weights(self, data):
        # Welch's method
        f, psd = signal.welch(data, fs=self.fs, nperseg=256)
        weights = 1.0 / np.sqrt(psd + 0.001)
        weights = weights / np.sqrt(np.mean(weights**2))
        self.weights = weights
        return weights
    
    def add_noise(self, gradient):
        if self.weights is None:
            raise ValueError("Run compute_weights first")
        
        # Clip
        norm = torch.norm(gradient)
        if norm > self.C:
            gradient = gradient * (self.C / norm)
        
        # FFT
        grad_fft = torch.fft.rfft(gradient)
        
        # Noise
        noise = torch.randn_like(grad_fft) * 0.1
        
        # Apply weights
        w_min = np.min(self.weights)
        sigma = (2 * np.sqrt(2 * np.log(1.25/self.delta))) / (self.epsilon * w_min)
        
        grad_fft = grad_fft + sigma * self.C * noise
        
        return torch.fft.irfft(grad_fft, n=len(gradient))
    
    def train(self, model, loader, epochs=10, lr=0.001):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()
                out = model(x)
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward()
                
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = self.add_noise(param.grad)
                
                optimizer.step()
