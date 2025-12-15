import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# SNN Components
class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron"""
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau  # membrane time constant
        self.vth = 0.5  # spike threshold - LOWERED for easier firing
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
    """Fast sigmoid surrogate for spike gradient"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Faster surrogate: clip-based instead of sigmoid
        grad_x = grad_output / (1 + torch.abs(x))
        return grad_x

class SNNLayer(nn.Module):
    """Spiking neural network layer with LIF neurons"""
    def __init__(self, in_features, out_features, num_steps=10, tau=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_steps = num_steps
        self.tau = tau
        
        # Learnable weights - initialize LARGER for better gradient flow
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Neuron state
        self.lif = LIFNeuron(tau=tau)
        
        # Statistics tracking
        self.spike_stats = {'total_spikes': 0, 'total_neurons': 0}
    
    def forward(self, spike_input):
        # spike_input: (batch, seq_len, in_features, num_steps)
        batch_size, seq_len, in_features, num_steps = spike_input.shape
        
        output_spikes = []
        
        self.hidden_spikes = None
        
        # Simulate across timesteps
        for t in range(num_steps):
            # Get input spikes at this timestep
            x_t = spike_input[:, :, :, t]  # (batch, seq_len, in_features)
            
            if self.hidden_spikes is not None:
                x_t = x_t + self.hidden_spikes * 0.1
            
            
            # Reshape for matrix mult
            x_t_flat = x_t.reshape(-1, in_features)
            
            # Compute input current: x = spike_input @ weight + bias
            i_t = torch.matmul(x_t_flat, self.weight) + self.bias
            i_t = i_t.reshape(batch_size, seq_len, self.out_features)
            
            # LIF neuron dynamics
            spike_out, _ = self.lif(i_t) 
            if t < num_steps - 5:
                spike_out = spike_out.detach()
            
            self.hidden_spikes = spike_out
            output_spikes.append(spike_out.unsqueeze(-1))
        
        # Stack across time: (batch, seq_len, out_features, num_steps)
        output = torch.cat(output_spikes, dim=-1)
        
        # Track spike statistics
        total_spikes = output.sum().item()
        total_neurons = output.numel()
        self.spike_stats['total_spikes'] = total_spikes
        self.spike_stats['total_neurons'] = total_neurons
        
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
        self.output_weight = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.5)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
    
    def encode_to_spikes(self, tokens):
        # tokens: (batch, seq_len)
        
        # Get embeddings
        embedded = self.embedding(tokens)  # (batch, seq_len, hidden_size)
        
        # Generate spike train using temporal/population coding
        # Use a steeper sigmoid to create more pronounced spike probabilities
        spike_probs = torch.sigmoid(embedded * 3.0)  # Increased multiplier for steeper probability curve
        
        # Create spike train
        spikes = []
        for t in range(self.num_steps):
            # Generate binary spike train by comparing random values to sigmoid probabilities
            spike_t = (torch.rand_like(embedded) < spike_probs).float()
            spikes.append(spike_t.unsqueeze(-1))
        
        spikes = torch.cat(spikes, dim=-1)
        # (batch, seq_len, hidden_size, num_steps)
        
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
    
    def get_spike_stats(self):
        """Get spike firing statistics from all layers"""
        stats = []
        for i, layer in enumerate(self.snn_layers):
            spikes = layer.spike_stats['total_spikes']
            neurons = layer.spike_stats['total_neurons']
            firing_rate = (spikes / neurons * 100) if neurons > 0 else 0
            stats.append({
                'layer': i,
                'total_spikes': int(spikes),
                'total_neurons': neurons,
                'firing_rate_%': firing_rate
            })
        return stats
    
    @torch.no_grad()
    def generate(self, prompt_tokens, max_length=50, temperature=1.0):
        """Autoregressive generation"""
        self.eval()
        device = next(self.parameters()).device
        
        # Handle both tensor and list inputs
        if isinstance(prompt_tokens, torch.Tensor):
            if prompt_tokens.dim() > 1:
                generated = prompt_tokens.squeeze().tolist()
            else:
                generated = prompt_tokens.tolist()
        else:
            generated = list(prompt_tokens)
        
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

def build_char_vocab(text):
    """Build character-level vocabulary from text"""
    vocab = sorted(set(text))
    token_to_id = {char: idx for idx, char in enumerate(vocab)}
    id_to_token = {idx: char for char, idx in token_to_id.items()}
    return token_to_id, id_to_token, len(vocab)

def encode_text(text, token_to_id):
    """Convert text to token IDs"""
    return [token_to_id.get(char, 0) for char in text]

def decode_tokens(tokens, id_to_token):
    """Convert token IDs back to text"""
    return ''.join([id_to_token.get(t, '?') for t in tokens])

def create_sequences(text, token_to_id, seq_len=16):
    """Create input/target pairs for training"""
    tokens = encode_text(text, token_to_id)
    
    inputs = []
    targets = []
    
    # Sliding window: input is tokens[i:i+seq_len], target is tokens[i+1:i+seq_len+1]
    for i in range(len(tokens) - seq_len):
        inputs.append(tokens[i:i+seq_len])
        targets.append(tokens[i+1:i+seq_len+1])
    
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# Add spike regularization to loss
def sparsity_regularization(model, lambda_spike=0.01):
    reg_loss = 0
    for layer in model.snn_layers:
        # Encourage reasonable firing rates (not too sparse, not too dense)
        firing_rate = layer.spike_stats['total_spikes'] / layer.spike_stats['total_neurons']
        target_rate = 0.15  # 10% firing rate target
        reg_loss += lambda_spike * (firing_rate - target_rate) ** 2
    return reg_loss

# Add small noise to gradients to escape local minima
def add_gradient_noise(model, noise_std=0.001):
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_std
            param.grad += noise

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits = model(inputs)
        
        # Compute loss
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        add_gradient_noise(model, noise_std=0.001)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, val_loader, device):
    """Evaluate on validation data"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create character vocabulary from sample text
    sample_text = """
    In a field one summer's day a Grasshopper was hopping about, 
    chirping and singing to its heart's content. An Ant passed by, 
    bearing along with great toil an ear of corn he was taking to the nest.

    "Why not come and chat with me," said the Grasshopper, 
    "instead of toiling and moiling in that way?"

    "I am helping to lay up food for the winter," said the Ant, 
    "and recommend you to do the same."

    "Why bother about winter?" said the Grasshopper; 
    "we have got plenty of food at present." 
    But the Ant went on its way and continued its toil. BREAK"""
    
    # Create character vocabulary from sample text
    small_input = ["The quick brown fox jumps over the lazy dog. BREAK",
    "Hello world and welcome to machine learning. BREAK",
    "Spiking neural networks are bio-inspired models. BREAK",
    "This model learns character level patterns in text. BREAK",
    "Each neuron has a membrane potential that integrates incoming spikes. When the potential crosses a threshold, the neuron fires a spike and the potential resets. BREAK",
    "Incoming spikes from presynaptic neurons cause changes in the membrane potential of the postsynaptic neuron. The effect of each spike is weighted by the synaptic strength between the neurons. BREAK",
    "Set the number of time steps and the sizes of input, hidden, and output layers. BREAK",
    "Initialize neurons and synapses with their parameters and random weights. BREAK",
    "Run the simulation for the defined number of time steps. BREAK",
    "Update neurons and synapses at each time step. BREAK",
    "Update neurons and synapses at each time step. BREAK",
    "Check if the pattern is detected. BREAK",
    "As the simulation progresses, the synaptic weights are adjusted based on the STDP rule, potentially making it more likely for the pattern to be detected if the network learns to recognize it. BREAK",
    "The leaky integrate-and-fire model with refractory periods and decay factors can cause neurons to spike at specific intervals, contributing to the pattern detection. BREAK",
    "SNNs excel at processing temporal information, as they naturally encode and process time through the timing of spikes. This capability is crucial for applications like speech recognition, time-series prediction, and dynamic sensory processing. BREAK",
    "raining SNNs is more challenging compared to traditional ANNs due to the discrete and non-differentiable nature of spikes. Researchers have developed various approaches, such as converting trained ANNs to SNNs and using surrogate gradient methods to overcome this hurdle. BREAK",
    "SNNs' ability to process sensory information in real-time makes them ideal for robotic applications, including autonomous navigation, sensorimotor control, and adaptive behavior. BREAK",
    "Spiking Neural Networks represent a significant leap towards more efficient and biologically plausible artificial intelligence. While challenges remain, the potential applications in neuromorphic computing, robotics, and brain-computer interfaces make SNNs a promising avenue for future research and development. As we continue to unravel the complexities of the human brain, SNNs will play a crucial role in bridging the gap between biological and artificial intelligence. BREAK",
    "While the analysis provides comprehensive insights into neuron models, training paradigms, and performance metrics, several limitations must be acknowledged. First, the evaluation relies primarily on benchmark datasets, which may not fully capture real-world complexity or large-scale deployment scenarios. BREAK",
    "Third, hyperparameter sensitivity in surrogate-gradient training and convergence instability in STDP highlight ongoing challenges that require further exploration. Finally, while comparative metrics such as accuracy, latency, and energy were integrated, additional factors such as scalability on high-dimensional tasks and robustness under noisy conditions remain areas for future research. Recognizing these limitations underscores that the findings, while promising, represent one step toward advancing brain-inspired and low-power AI systems. BREAK"
    ]
    for entry in small_input:
        sample_text += entry
    
    sample_text += """Spiking Neural Networks are a class of artificial neural networks that mimic the behavior of biological neurons more closely than traditional neural networks. In SNNs, neurons communicate by sending discrete spikes, which represent changes in voltage across a neuron's membrane. These spikes are generated when the membrane potential exceeds a certain threshold.

    The human brain consists of approximately 86 billion neurons, which communicate through electrical impulses known as action potentials or spikes. This communication method is energy-efficient and highly effective for processing information. SNNs aim to replicate this spiking behavior, leveraging the brain's mechanisms for computation and learning. BREAK"""
    token_to_id, id_to_token, vocab_size = build_char_vocab(sample_text)
    
    print(f"Vocabulary: {list(token_to_id.keys())}")
    print(f"Vocab size: {vocab_size}\n")
    
    # Hyperparameters
    hidden_size = 64
    num_layers = 2
    num_steps = 20  # Timesteps per token
    seq_len = 16
    batch_size = 8
    learning_rate = 0.0001  # Even lower for stochastic training
    num_epochs = 100
    
    # Create model with actual vocab size
    model = SNNTextModel(vocab_size, hidden_size, num_layers, num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # More aggressive scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=6,
        min_lr=1e-6
    )
    
    
    # Create training data
    inputs, targets = create_sequences(sample_text, token_to_id, seq_len)
    print(f"Created {len(inputs)} training sequences\n")
    
    # Split into train/val (80/20)
    split_idx = int(len(inputs) * 0.8)
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # Training loop
    print("=" * 50)
    print("TRAINING")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Print spike statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            stats = model.get_spike_stats()
            print("Spike Statistics:")
            for stat in stats:
                firing_rate = stat['firing_rate_%']
                print(f"  Layer {stat['layer']}: {firing_rate:.2f}% firing ({stat['total_spikes']}/{stat['total_neurons']} spikes)")
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Generate sample every epoch to see progress
        prompt_text = "the "
        prompt_tokens = encode_text(prompt_text, token_to_id)
        generated_tokens = model.generate(prompt_tokens, max_length=30)
        generated_text = decode_tokens(generated_tokens, id_to_token)
        print(f"Sample: '{generated_text}'")
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)