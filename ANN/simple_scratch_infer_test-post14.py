import torch
import random
import numpy as np
import matplotlib.pyplot as plt 
import time
import ftfy
import re
from prettytable import PrettyTable
from collections import defaultdict

class SimpleBPETokenizer:
    def __init__(self):
        self.vocab = []
        self.merges = {}
        self.vocab_size = 0
        self.dictionary = {}
        self.reverse_dictionary = {}
        
    def __add_to_dict(self, character):
        if character not in self.dictionary:
            self.dictionary[character] = len(self.dictionary)
            self.reverse_dictionary[self.dictionary[character]] = character
    
    def character_to_token(self, character):
        return self.dictionary[character]

    def token_to_character(self, token):
        return self.reverse_dictionary[token]
    
    def size(self):
        return len(self.dictionary)
    
    def pre_tokenize(self, text):
        """Split text into words and handle spaces properly"""
        # Add special space markers between words
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split(' ')
        return words
    
    def get_stats(self, splits):
        """Get frequency of all adjacent pairs"""
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def merge_pair(self, a, b, splits):
        """Merge pair (a, b) in all splits"""
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    def train(self, texts, vocab_size=300):
        """Train BPE tokenizer on given texts"""
        # Step 1: Build word frequencies with space markers
        self.word_freqs = defaultdict(int)
        
        for text in texts:
            text = re.sub(r'\s+', ' ', text.strip())
            words = text.split(' ')
            for i, word in enumerate(words):
                if not word:
                    continue
                # Add space marker for all words except the first one
                if i > 0:
                    word_with_space = 'Ġ' + word
                else:
                    word_with_space = word
                self.word_freqs[word_with_space] += 1
        
        # Build initial alphabet
        alphabet = set()
        for word in self.word_freqs.keys():
            for char in word:
                alphabet.add(char)
        alphabet = sorted(alphabet)
        
        # Initial vocabulary: special tokens + alphabet
        self.vocab = ["<EOS>", "<BOS>", "<PAD>", "<UNK>"] + alphabet
        self.merges = {}
        
        # Initial splits: each word split into characters
        splits = {word: [c for c in word] for word in self.word_freqs.keys()}
        
        # Step 2: Perform BPE merges
        while len(self.vocab) < vocab_size:
            pair_freqs = self.get_stats(splits)
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            max_freq = pair_freqs[best_pair]
            
            if max_freq == 0:
                break
                
            # Merge the best pair
            splits = self.merge_pair(*best_pair, splits)
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)
            
            # print(f"Merged {best_pair} -> '{merged_token}', vocab size: {len(self.vocab)}")
            
            # Early stopping if we're not making progress
            if len(self.vocab) % 50 == 0 and len(self.merges) < 10:
                print("Stopping early - not enough merges happening")
                break
        
        self.vocab_size = len(self.vocab)
        for word in self.vocab:
            self.__add_to_dict(word)
        
        return self.vocab
    
    def tokenize(self, text):
        """Tokenize text into subword tokens"""
        # Pre-tokenize into words with space markers
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split(' ')
        
        tokens = []
        for i, word in enumerate(words):
            if not word:
                continue
                
            # Add space marker for all words except the first one
            if i > 0:
                word_to_tokenize = 'Ġ' + word
            else:
                word_to_tokenize = word
            
            # Start with individual characters
            split = list(word_to_tokenize)
            
            # Apply all learned merges
            changed = True
            while changed and len(split) > 1:
                changed = False
                i = 0
                while i < len(split) - 1:
                    pair = (split[i], split[i + 1])
                    if pair in self.merges:
                        split = split[:i] + [self.merges[pair]] + split[i + 2:]
                        changed = True
                    else:
                        i += 1
            
            # Convert to token IDs
            for token in split:
                if token in self.vocab:
                    tokens.append(self.vocab.index(token))
                else:
                    tokens.append(self.vocab.index("<UNK>"))
        
        return tokens
    
    def detokenize(self, token_ids):
        """Convert token IDs back to text"""
        tokens = [self.vocab[i] for i in token_ids]
        
        text = ""
        for token in tokens:
            if token.startswith('Ġ'):
                if text:  # Only add space if we already have text
                    text += ' ' + token[1:]
                else:
                    text += token[1:]  # First word, no space
            else:
                text += token
        
        return text         

class TokenEmbedding(torch.nn.Module):
    """
    PyTorch module that converts tokens into embeddings.

    Input dimension is: (batch_size, sequence_length)
    Output dimension is: (batch_size, sequence_length, d_model)
    """

    def __init__(self, d_model, number_of_tokens):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=number_of_tokens,
            embedding_dim=d_model
        ).to(device)

    def forward(self, x):
        return self.embedding_layer(x)        


class PositionalEncoding(torch.nn.Module):
    """
    Pytorch module that creates a positional encoding matrix. This matrix will later be added to the 
    transformer's input embeddings to provide a sense of position of the sequence elements.
    """

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        """
        Creates a positional encoding matrix of size (max_sequence_length, d_model).
        """

        # Initialize positional encoding matrix
        positional_encoding = np.zeros((self.max_sequence_length, self.d_model))

        # Calculate positional encoding for each position and each dimension
        for pos in range(self.max_sequence_length):
            for i in range(0, self.d_model, 2):
                # Apply sin to even indices in the array; indices in Python start at 0 so i is even.
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                
                if i + 1 < self.d_model:
                    # Apply cos to odd indices in the array; we add 1 to i because indices in Python start at 0.
                    positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / self.d_model)))

        # Convert numpy array to PyTorch tensor and return it
        return torch.from_numpy(positional_encoding).to(device).float()

    def forward(self, x):
        """
        Adds the positional encoding to the input embeddings at the corresponding positions.
        """
        # Add positional encodings to input embeddings. The ":" indexing ensures we only add positional encodings up
        # to the length of the sequence in the batch. x.size(0) is the batch size, so this is a way to make sure 
        # we're not adding extra positional encodings.
        return x + self.positional_encoding[:x.size(1), :]
        
        
class MaskedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a self attention layer.
    This layer is used in the MultiHeadedSelfAttention module.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, head_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.query_layer = torch.nn.Linear(embedding_dimension, self.head_dimension).to(device)
        self.key_layer = torch.nn.Linear(embedding_dimension, self.head_dimension).to(device)
        self.value_layer = torch.nn.Linear(embedding_dimension, self.head_dimension).to(device)
        self.softmax = torch.nn.Softmax(dim=-1).to(device)

    def forward(self, x, mask):
        """
        Compute the self attention.

        x dimension is: (batch_size, sequence_length, embedding_dimension)
        output dimension is: (batch_size, sequence_length, head_dimension)
        mask dimension is: (batch_size, sequence_length)

        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # x dimensions are: (batch_size, sequence_length, embedding_dimension)
        # query, key, value dimensions are: (batch_size, sequence_length, head_dimension)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Calculate the attention weights.
        # attention_weights dimensions are: (batch_size, sequence_length, sequence_length)
        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        # Scale the attention weights.
        attention_weights = attention_weights / np.sqrt(self.head_dimension)

        # Apply the mask to the attention weights, by setting the masked tokens to a very low value.
        # This will make the softmax output 0 for these values.
        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
        # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
        attention_scores = self.softmax(attention_weights)

        # The attention scores are multiplied by the value
        # Values of tokens with high attention score get highlighted because they are multiplied by a larger number,
        # and tokens with low attention score get drowned out because they are multiplied by a smaller number.
        # Output dimensions are: (batch_size, sequence_length, head_dimension)
        return torch.bmm(attention_scores, value)
        
        
class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a multi head attention layer.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)]).to(device)

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension).to(device)

    def forward(self, x, mask):
        """
        Compute the multi head attention.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the self attention for each head
        # self_attention_outputs dimensions are:
        # (number_of_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, number_of_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output dimensions are: (batch_size, sequence_length, embedding_dimension)
        return self.output_layer(concatenated_self_attention_outputs)
        
class DecoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        
        # Pre-normalization (more stable training)
        self.norm1 = torch.nn.LayerNorm(embedding_dimension).to(device)
        self.norm2 = torch.nn.LayerNorm(embedding_dimension).to(device)
        
        self.self_attention = MaskedMultiHeadedSelfAttention(embedding_dimension, number_of_heads)
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
        
        self.dropout = torch.nn.Dropout(dropout_rate).to(device)

    def forward(self, x, mask):
        # Pre-normalization: Layer normalization BEFORE the operation
        normalized_x = self.norm1(x)  # Norm first
        attention_output = self.self_attention(normalized_x, mask)
        x = x + self.dropout(attention_output)  # Residual connection
        
        normalized_x = self.norm2(x)  # Norm first  
        ff_output = self.feed_forward(normalized_x)
        x = x + self.dropout(ff_output)  # Residual connection
        return x
        
class DecoderStack(torch.nn.Module):
    """
    Pytorch module for a stack of decoders.
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate) for _ in
             range(number_of_layers)]).to(device)

    def forward(self, x, mask):
        decoder_outputs = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs, mask)

        return decoder_outputs
        
        
class FeedForward(torch.nn.Module):
    """
    Pytorch module for a feed forward layer.

    A feed forward layer is a fully connected layer with a ReLU activation function in between.
    """

    def __init__(self, embedding_dimension, feed_forward_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension).to(device)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension).to(device)

    def forward(self, x):
        """
        Compute the feed forward layer.
        """
        return self.linear_2(torch.relu(self.linear_1(x)))
        
        
class LanguageModel(torch.nn.Module):
    """
    Pytorch module for a language model.
    """

    def __init__(
            self,
            number_of_tokens,  # The number of tokens in the vocabulary
            max_sequence_length=512,  # The maximum sequence length to use for attention
            embedding_dimension=512,  # The dimension of the token embeddings
            number_of_layers=6,  # The number of decoder layers to use
            number_of_heads=4,  # The number of attention heads to use
            feed_forward_dimension=None,  # The dimension of the feed forward layer
            dropout_rate=0.1  # The dropout rate to use
    ):
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        if feed_forward_dimension is None:
            # GPT-2 paper uses 4 * embedding_dimension for the feed forward dimension
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension

        self.dropout_rate = dropout_rate

        # Create the token embedding layer
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens).to(device)

        # Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_sequence_length)

        self.embedding_dropout = torch.nn.Dropout(dropout_rate).to(device)

        # Create the decoder stack
        self.decoder = DecoderStack(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            feed_forward_dimension=self.feed_forward_dimension,
            dropout_rate=dropout_rate,
            max_sequence_length=self.max_sequence_length
        ).to(device)
        
        self.final_layer_norm  = torch.nn.LayerNorm(embedding_dimension).to(device)

        # Create the language model head
        self.lm_head = torch.nn.Linear(embedding_dimension, number_of_tokens, bias=False).to(device)
        self.lm_head.weight = self.token_embedding.embedding_layer.weight
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, mask):
        # Embed tokens
        token_embeddings = self.token_embedding(x)
        
        # Scale embeddings (important for transformer stability)
        token_embeddings = token_embeddings * (self.embedding_dimension ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(token_embeddings)
        x = self.embedding_dropout(x)
        
        # Decoder
        x = self.decoder(x, mask)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        # LM head
        return self.lm_head(x)
        
    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'number_of_tokens': self.number_of_tokens,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dimension': self.embedding_dimension,
            'number_of_layers': self.number_of_layers,
            'number_of_heads': self.number_of_heads,
            'feed_forward_dimension': self.feed_forward_dimension,
            'dropout_rate': self.dropout_rate,
            'model_state_dict': self.state_dict()
        }, path)

    @staticmethod
    def load_checkpoint(path) -> 'LanguageModel':
        checkpoint = torch.load(path)
        model = LanguageModel(
            number_of_tokens=checkpoint['number_of_tokens'],
            max_sequence_length=checkpoint['max_sequence_length'],
            embedding_dimension=checkpoint['embedding_dimension'],
            number_of_layers=checkpoint['number_of_layers'],
            number_of_heads=checkpoint['number_of_heads'],
            feed_forward_dimension=checkpoint['feed_forward_dimension'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
        
class LMHead(torch.nn.Module):
    """
    Pytorch module for the language model head.
    The language model head is a linear layer that maps the embedding dimension to the vocabulary size.
    """

    def __init__(self, embedding_dimension, number_of_tokens):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens).to(device)

    def forward(self, x):
        """
        Compute the language model head.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        output dimensions are: (batch_size, sequence_length, number_of_tokens)
        """
        # Compute the linear layer
        # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)

        return linear_output
        
class AutoregressiveWrapper(torch.nn.Module):
    """
    Pytorch module that wraps a GPT model and makes it autoregressive.
    """

    def __init__(self, gpt_model):
        super().__init__()
        self.model = gpt_model
        self.max_sequence_length = self.model.max_sequence_length

    def forward(self, x, mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]

        output = self.model(inp, mask).to(device)
        return output, target

    def next_token_probabilities(self, x, mask, temperature=1.0):
        """
        Calculate the token probabilities for the next token in the sequence.
        """
        logits = self.model(x, mask)[:, -1]

        # Apply the temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply the softmax
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities
        
    def save_checkpoint(self, path):
        self.model.save_checkpoint(path)

    @staticmethod
    def load_checkpoint(path) -> 'AutoregressiveWrapper':
        model = LanguageModel.load_checkpoint(path)
        return AutoregressiveWrapper(model)
        
class Trainer:

    def __init__(self, model, tokenizer: SimpleBPETokenizer, optimizer=None):
        super().__init__()
        self.model = model
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.character_to_token('<pad>')).to(device)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100//4)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
    def create_batches(self, data, batch_size):
        """Pre-create all batches for efficiency"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch_sequences = data[i:i + batch_size]
            
            # Find max length in this batch for dynamic padding
            max_len = max(len(seq) for seq in batch_sequences)
            
            sequence_tensor = torch.full((len(batch_sequences), max_len), 
                                       self.tokenizer.character_to_token('<pad>'), 
                                       dtype=torch.long).to(device)
            mask_tensor = torch.zeros((len(batch_sequences), max_len), 
                                    dtype=torch.long).to(device)
            
            for j, seq in enumerate(batch_sequences):
                seq_len = len(seq)
                sequence_tensor[j, :seq_len] = torch.tensor(seq, dtype=torch.long)
                mask_tensor[j, :seq_len] = 1
            
            batches.append((sequence_tensor, mask_tensor))
        return batches

    def train(self, data: list[str], epochs, batch_size):
        loss_per_epoch = []
        
        # Pre-create all batches
        print("Creating batches...")
        batches = self.create_batches(data, batch_size)
        print(f"Created {len(batches)} batches")
        
        for epoch in range(epochs):
            startTime = time.time()
            losses = []

            # Shuffle batches instead of data
            random.shuffle(batches)

            for sequence_tensor, mask_tensor in batches:
                self.model.train()
                self.optimizer.zero_grad()

                # Input is all but last token, target is all but first
                input_tensor = sequence_tensor[:, :-1]
                target_tensor = sequence_tensor[:, 1:]
                input_mask = mask_tensor[:, :-1]

                # Forward pass
                model_output, target = self.model.forward(x=input_tensor, mask=input_mask)
                
                # Compute loss (ignore padding tokens)
                loss = self.loss_function(model_output.reshape(-1, model_output.size(-1)), 
                                        target.reshape(-1))

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                losses.append(loss.item())

            self.scheduler.step()

            epoch_loss = np.average(losses)
            loss_per_epoch.append(epoch_loss)
            print(f'Epoch: {epoch}, Time: {time.time()-startTime:.2f}s, Loss: {epoch_loss:.6f}')
            print(f'\tLR: {self.scheduler.get_last_lr()[0]:.2e}')

        plt.plot(loss_per_epoch)
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        return loss_per_epoch


def pad_left(sequence, final_length, padding_token):
    return [padding_token] * (final_length - len(sequence)) + sequence


class Generator:

    def __init__(
            self,
            model,
            tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
            self,
            max_tokens_to_generate: int,
            prompt: str = None,
            temperature: float = 1.0,
            eos_token: int = None,
            padding_token: int = 0):

        self.model.eval()

        if prompt is None:
            start_tokens = [self.tokenizer.character_to_token(padding_token)]
        else:
            start_tokens = self.tokenizer.tokenize(prompt)

        input_tensor = torch.tensor(
            pad_left(
                sequence=start_tokens,
                final_length=self.model.max_sequence_length + 1,
                padding_token=padding_token
            ),
            dtype=torch.long
        ).to(device)

        num_dims = len(input_tensor.shape)

        if num_dims == 1:
            input_tensor = input_tensor[None, :]

        out = input_tensor
        for _ in range(max_tokens_to_generate):

            x = out[:, -self.model.max_sequence_length:]

            mask = torch.ones_like(x)
            mask[x == padding_token] = 0

            # Compute the next token probabilities
            next_token_probabilities = self.model.next_token_probabilities(
                x=x,
                temperature=temperature,
                mask=mask
            )

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(next_token_probabilities, num_samples=1)

            # Append the next token to the output
            out = torch.cat([out, next_token], dim=1)

            # If the end of sequence token is reached, stop generating tokens
            if eos_token is not None and next_token == eos_token:
                break

        generated_tokens = out[0].tolist()
        return self.tokenizer.detokenize(generated_tokens)

def create_training_sequences(max_sequence_length, tokenized_training_data):
    # Create sequences of length max_sequence_length + 1
    # The last token of each sequence is the target token
    sequences = []
    for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
        sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])
    return sequences


def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    # Tokenize the training data
    tokenized_training_data = tokenizer.tokenize(training_data)
    for _ in range(max_sequence_length):
        # Prepend padding tokens
        tokenized_training_data.insert(0, tokenizer.character_to_token('<pad>'))
    return tokenized_training_data


tokenizer = SimpleBPETokenizer()
training_data = [
    'cats rule the world.',
    'dogs are the best.', 
    'elephants have long trunks.',
    'monkeys like bananas.',
    'pandas eat bamboo.',
    'tigers are dangerous.',
    'zebras have stripes.',
    'lions are the kings of the savannah.',
    'giraffes have long necks.',
    'hippos are big and scary.',
    'rhinos have horns.',
    'penguins live in the arctic.',
    'polar bears are white.'
]

# Add your file data
files_to_load = [
    "data/short_snippets.txt",
    "data/sierra_data.txt", 
    "data/forsaken_data.txt",
    "data/dominion_rp_epd.txt",
    "data/dominion_rp_disestro.txt",
    "data/dominion_rp_disestro.txt"
]

for file_path in files_to_load:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            training_data.append(content)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        continue

# Clean the data
training_data = [ftfy.fix_text(text) for text in training_data]
vocab = tokenizer.train(training_data, vocab_size=1000)
print(vocab)
print("Vocab Size: ", tokenizer.size())

embedding_dimension = 256
number_of_tokens = tokenizer.size()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create the model
model = AutoregressiveWrapper(LanguageModel(
    embedding_dimension=embedding_dimension,
    number_of_tokens=number_of_tokens,
    number_of_heads=4,
    number_of_layers=3,
    dropout_rate=0.1,
    max_sequence_length=60
).to(device)).to(device)
model = model.load_checkpoint("models/text_completion_17_85_of_100.pth")
model.eval()
max_tokens_to_generate = 60
generator = Generator(model, tokenizer)



while True:
    promptStr = input("Prompt: ")
    generated_text = generator.generate(
        max_tokens_to_generate=max_tokens_to_generate,
        prompt=promptStr,
        padding_token=tokenizer.character_to_token('<PAD>')
    )
    generated_text = generated_text.replace('<PAD>', '')
    print(generated_text)





    



#training_data = ['cats rule the world',
#    'dogs are the best',
#    'elephants have long trunks',
#    'monkeys like bananas',
#    'pandas eat bamboo',
#    'tigers are dangerous',
#    'zebras have stripes',
#    'lions are the kings of the savannah',
#    'giraffes have long necks',
#    'hippos are big and scary',
#    'rhinos have horns',
#    'penguins live in the arctic',
#    'polar bears are white']
#
#tokenized_data = []
#max_seq_len = 0
#used_tokenizer = Tokenizer()
#for entry in training_data:
#    tokenized_entry = used_tokenizer.tokenize(entry)
#    max_seq_len = max(len(tokenized_entry), max_seq_len)
#    tokenized_data.append(tokenized_entry)
#
#padded_training_data = []
#for entry in tokenized_data:
#    new_entry = entry[:]
#    for _ in range(len(entry), max_seq_len):
#        new_entry.insert(0, used_tokenizer.character_to_token('<pad>'))
#    padded_training_data.append(new_entry)
#print(padded_training_data)