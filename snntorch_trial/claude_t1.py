import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import spikegen

class SpikingTextPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, beta=0.9):
        """
        Create a spiking neural network for text prediction
        
        Args:
            input_size (int): Size of the input vocabulary
            hidden_size (int): Number of hidden neurons
            output_size (int): Size of the output vocabulary
            num_layers (int): Number of recurrent layers
            beta (float): Membrane potential decay rate
        """
        super().__init__()
        
        # Embedding layer to convert input characters to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Spiking recurrent layers using Leaky Integrate-and-Fire (LIF) neurons
        self.snn_layers = nn.ModuleList([
            snn.LIF(alpha=hidden_size, beta=beta) for _ in range(num_layers)
        ])
        
        # Output layer to convert spikes to predictions
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the spiking neural network
        
        Args:
            x (torch.Tensor): Input sequence of characters
            hidden (tuple, optional): Previous hidden state
        
        Returns:
            torch.Tensor: Output predictions
            tuple: Updated hidden state
        """
        # Embed input characters
        x = self.embedding(x)
        
        # Process through spiking layers
        for layer in self.snn_layers:
            spk, x = layer(x)
        
        # Convert spikes to output predictions
        output = self.fc(x)
        
        return output


def prepare_sequence(text, char_to_idx):
    """
    Convert text to tensor of character indices
    
    Args:
        text (str): Input text sequence
        char_to_idx (dict): Mapping of characters to indices
    
    Returns:
        torch.Tensor: Sequence of character indices
    """
    return torch.tensor([char_to_idx[char] for char in text], dtype=torch.long)

def train_snn_text_predictor(model, text, char_to_idx, idx_to_char, num_epochs=100):
    """
    Train the spiking neural network for text prediction
    
    Args:
        model (SpikingTextPredictor): Spiking neural network model
        text (str): Training text
        char_to_idx (dict): Character to index mapping
        idx_to_char (dict): Index to character mapping
        num_epochs (int): Number of training epochs
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Create input and target sequences
        input_seq = prepare_sequence(text[:-1], char_to_idx)
        target_seq = prepare_sequence(text[1:], char_to_idx)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_seq)
        
        # Compute loss
        loss = criterion(output, target_seq)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print loss periodically
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def generate_text(model, seed_text, char_to_idx, idx_to_char, length=50):
    """
    Generate text using the trained spiking neural network
    
    Args:
        model (SpikingTextPredictor): Trained spiking neural network
        seed_text (str): Initial text to start generation
        char_to_idx (dict): Character to index mapping
        idx_to_char (dict): Index to character mapping
        length (int): Number of characters to generate
    
    Returns:
        str: Generated text
    """
    model.eval()  # Set the model to evaluation mode
    current_text = seed_text
    
    with torch.no_grad():
        for _ in range(length):
            # Convert current text to input sequence
            input_seq = prepare_sequence(current_text[-1], char_to_idx).unsqueeze(0)
            
            # Generate prediction
            output = model(input_seq)
            
            # Get the most likely next character
            _, predicted_idx = torch.max(output, dim=1)
            predicted_char = idx_to_char[predicted_idx.item()]
            
            # Append the predicted character
            current_text += predicted_char
    
    return current_text

def main():
    # Example training text (you can replace with a larger corpus)
    training_text = "Hello, this is a sample text for training a spiking neural network for text prediction."
    
    # Create character mappings
    unique_chars = sorted(list(set(training_text)))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    # Model hyperparameters
    input_size = len(unique_chars)
    hidden_size = 64
    output_size = len(unique_chars)
    
    # Initialize the spiking neural network model
    model = SpikingTextPredictor(
        input_size=input_size, 
        hidden_size=hidden_size, 
        output_size=output_size
    )
    
    # Train the model
    train_snn_text_predictor(
        model, 
        training_text, 
        char_to_idx, 
        idx_to_char
    )
    
    # Generate text
    seed_text = training_text[:5]  # Use first 5 characters as seed
    generated_text = generate_text(
        model, 
        seed_text, 
        char_to_idx, 
        idx_to_char
    )
    
    print("\nGenerated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()





