import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

# Leaky neuron model, overriding the backward pass with a custom function
class LeakySurrogate(nn.Module):
  def __init__(self, beta, threshold=1.0):
      super(LeakySurrogate, self).__init__()

      # initialize decay rate beta and threshold
      self.beta = beta
      self.threshold = threshold
      self.spike_gradient = self.ATan.apply

  # the forward function is called each time we call Leaky
  def forward(self, input_, mem):
    spk = self.spike_gradient((mem-self.threshold))  # call the Heaviside function
    reset = (self.beta * spk * self.threshold).detach()  # remove reset from computational graph
    mem = self.beta * mem + input_ - reset  # Eq (1)
    return spk, mem

  # Forward pass: Heaviside function
  # Backward pass: Override Dirac Delta with the derivative of the ArcTan function
  @staticmethod
  class ATan(torch.autograd.Function):
      @staticmethod
      def forward(ctx, mem):
          spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
          ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
          return spk

      @staticmethod
      def backward(ctx, grad_output):
          (spk,) = ctx.saved_tensors  # retrieve the membrane potential
          grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
          return grad
		  

# dataloader arguments
batch_size = 128
data_path='/tmp/data/mnist'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

def build_char_vocab(text_array):
    vocab = set()
    vocab.add("<UNK>")
    vocab.add("<PAD>")
    vocab.add("<BOS>")
    vocab.add("<EOS>")
    vocab.add(" ")
    longest_len = 0
    for entry in text_array:
        sentence = entry.split(" ")
        longest_len = max(longest_len, len(sentence))
        for word in sentence:
            vocab.add(word)
    
    token_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_token = {idx: word for word, idx in token_to_id.items()}
    return token_to_id, id_to_token, len(vocab), longest_len

def encode_text(text, token_to_id, long_len):
    """Convert text to token IDs"""
    sentence = text.split(" ")
    ec_text = []
    sentence_len = len(sentence)
    ec_text.append(token_to_id.get("<BOS>"))
    for index in range(sentence_len):
        if sentence[index] not in token_to_id:
            # print(f"{sentence[index]} is not a known token")
            ec_text.append(token_to_id.get("<UNK>"))
        else:
            # print(f"{sentence[index]} append as a token")
            ec_text.append(token_to_id.get(sentence[index]))
        if index < sentence_len - 1:
            # print(f"Appending a space")
            ec_text.append(token_to_id.get(" "))
            
    ec_text.append(token_to_id.get("<EOS>"))
    while len(ec_text) < long_len:
        # print(f"Appending a PAD")
        ec_text.append(token_to_id.get("<PAD>"))

    return ec_text

def decode_tokens(tokens, id_to_token):
    """Convert token IDs back to text"""
    return ''.join([id_to_token.get(t, '?') for t in tokens])

def create_sequences(text, token_to_id, longest_len, seq_len=12):
    """Create input/target pairs for training"""
    inputs = []
    targets = []
    
    for entry in text:
        tokens = encode_text(entry, token_to_id, longest_len)
        while len(tokens) <= seq_len:
            tokens.append(token_to_id.get("<PAD>"))
        
        # Sliding window: input is tokens[i:i+seq_len], target is tokens[i+1:i+seq_len+1]
        for i in range(len(tokens) - seq_len):
            inputs.append(tokens[i:i+seq_len])
            targets.append(tokens[i+1:i+seq_len+1])
    
    return inputs, targets
    
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
token_to_id, id_to_token, vocab_size, longest_len = build_char_vocab(sample_text)

inputs = []
for entry in sample_text:









# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Load the network onto CUDA if available
net = Net().to(device)


# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(epoch, iter_counter, data, targets, test_data, loss_hist, counter, train):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")
	
	
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
num_epochs = 1
loss_hist = []
test_loss_hist = []
counter = 0

for epoch in range(num_epochs):
    iteration_counter = 0
    training_batch = iter(train_loader)
    
    for data, targets in training_batch:
        data = data.to(device)
        targets = targets.to(device)
        
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))
        
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        loss_hist.append(loss_val.item())
        
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                train_printer(epoch, iteration_counter, data, targets, test_data, loss_hist, counter, False)
            counter += 1
            iteration_counter +=1
            
# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()