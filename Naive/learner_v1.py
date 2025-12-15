import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# SNN Components
class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron"""
    def __init__(self, tau=0.5, vth=1.0):
        super().__init__()
        self.tau = tau  # membrane time constant
        self.vth = vth  # spike threshold
        self.v = None  # membrane voltage (will be created dynamically)
    
    def forward(self, x, dt=0.1):
        # x: input current to neuron (batch, seq_len, features)
        
        # Initialize voltage if needed
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x, device=x.device)
        
        # LIF dynamics: dv/dt = -v/tau + x
        self.v = self.v * (1 - dt / self.tau) + x * dt
        
        # Generate spikes
        spike = (self.v >= self.vth).float()
        
        # Reset voltage after spike
        self.v = self.v * (1 - spike)
        
        return spike, self.v

class SurrogateSpike(torch.autograd.Function):
    """Straight-through estimator for spike gradient"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Sigmoid derivative as surrogate
        grad_x = grad_output * torch.sigmoid(x) * (1 - torch.sigmoid(x))
        return grad_x

class SNNLayer(nn.Module):
    """Spiking neural network layer with LIF neurons"""
    def __init__(self, in_features, out_features, num_steps=10, tau=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_steps = num_steps
        self.tau = tau
        
        # Learnable weights
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Neuron state
        self.lif = LIFNeuron(tau=tau)
    
    def forward(self, spike_input):
        # spike_input: (batch, seq_len, in_features, num_steps)
        batch_size, seq_len, in_features, num_steps = spike_input.shape
        
        output_spikes = []
        
        # Simulate across timesteps
        for t in range(num_steps):
            # Get input spikes at this timestep
            x_t = spike_input[:, :, :, t]  # (batch, seq_len, in_features)
            
            # Reshape for matrix mult
            x_t_flat = x_t.reshape(-1, in_features)
            
            # Compute input current: x = spike_input @ weight + bias
            i_t = torch.matmul(x_t_flat, self.weight) + self.bias
            i_t = i_t.reshape(batch_size, seq_len, self.out_features)
            
            # LIF neuron dynamics
            spike_out, _ = self.lif(i_t)
            output_spikes.append(spike_out.unsqueeze(-1))
        
        # Stack across time: (batch, seq_len, out_features, num_steps)
        output = torch.cat(output_spikes, dim=-1)
        return output

class SNNTextModel(nn.Module):
    """SNN-based text-to-text model"""
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, num_steps=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        
        # Embedding: convert token indices to rate-coded spike patterns
        # We'll use a simple embedding then convert to spikes
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Spiking layers
        self.snn_layers = nn.ModuleList([
            SNNLayer(hidden_size, hidden_size, num_steps=num_steps)
            for _ in range(num_layers)
        ])
        
        # Output layer: decode spikes back to logits
        self.output_weight = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
    
    def encode_to_spikes(self, tokens):
        # tokens: (batch, seq_len)
        
        # Get embeddings
        embedded = self.embedding(tokens)  # (batch, seq_len, hidden_size)
        
        # Convert to spikes using rate coding
        # Spike probability proportional to embedding magnitude (normalized)
        spike_probs = torch.sigmoid(embedded)  # Map to [0,1]
        
        # Sample spikes for each timestep
        # Expand and repeat for num_steps: (batch, seq_len, hidden_size, num_steps)
        spike_probs_expanded = spike_probs.unsqueeze(-1).expand(-1, -1, -1, self.num_steps)
        spikes = torch.bernoulli(spike_probs_expanded)
        
        return spikes
    
    def forward(self, tokens):
        # Encode tokens to spike trains
        spike_input = self.encode_to_spikes(tokens)
        
        # Pass through SNN layers
        x = spike_input
        for snn_layer in self.snn_layers:
            x = snn_layer(x)
        
        # Decode spikes back to token logits
        # Sum spikes across time (rate coding): more spikes = stronger signal
        spike_counts = x.sum(dim=-1)  # (batch, seq_len, hidden_size)
        
        # Project to vocabulary
        logits = torch.matmul(spike_counts, self.output_weight) + self.output_bias
        # (batch, seq_len, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_tokens, max_length=50, temperature=1.0):
        """Autoregressive generation"""
        self.eval()
        device = next(self.parameters()).device
        
        # Ensure prompt is a flat list
        if prompt_tokens.dim() > 1:
            generated = prompt_tokens.squeeze().tolist()
        else:
            generated = prompt_tokens.tolist()
        
        # Make it a list if it's a single int
        if not isinstance(generated, list):
            generated = [generated]
        
        for _ in range(max_length):
            # Get logits for the sequence so far
            input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.forward(input_tensor)
            
            # Take last token's logits
            next_logits = logits[0, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            # Stop if EOS token (assuming vocab_size-1 is EOS)
            if next_token == self.vocab_size - 1:
                break
        
        return generated

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    vocab_size = 256  # Small vocab for testing
    hidden_size = 64
    num_layers = 2
    num_steps = 5  # Timesteps per token
    batch_size = 4
    seq_len = 8
    
    # Create model
    model = SNNTextModel(vocab_size, hidden_size, num_layers, num_steps).to(device)
    
    # Dummy input
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    print("Input shape:", tokens.shape)
    logits = model(tokens)
    print("Output logits shape:", logits.shape)
    print("Output vocab probs shape:", F.softmax(logits, dim=-1).shape)
    
    # Try generation
    prompt = torch.tensor([[1, 2, 3]], device=device)
    generated = model.generate(prompt, max_length=20)
    print("Generated tokens:", generated)
    
    # Simple loss (would use cross-entropy in real training)
    target = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    print(f"Loss: {loss.item():.4f}")