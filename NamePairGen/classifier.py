import questionary
from questionary import Choice, Style
from typing import Any, TypeVar
from pathlib import Path
import ftfy


def prompt_select(message: str, choices: list[Any]) -> Any:
    return questionary.select(
        message,
        choices=choices,
        style=Style([("highlighted", "reverse")]),
    ).ask()


def get_file_names(directory_path, sort=True, exclude_hidden=False):
    """
    Get file names from a directory with options.

    Args:
        directory_path (str): Path to the directory
        sort (bool): Whether to sort file names alphabetically
        exclude_hidden (bool): Whether to exclude hidden files (starting with .)

    Returns:
        list: List of file names
    """
    try:
        # Convert to Path object for better handling
        path = Path(directory_path)

        # Check if path exists
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")

        # Check if it's a directory
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")

        # Get all files
        files = []
        for item in path.iterdir():
            if item.is_file():
                file_name = item.name
                # Skip hidden files if requested
                if exclude_hidden and file_name.startswith('.'):
                    continue
                files.append(file_name)

        # Sort if requested
        if sort:
            files.sort()

        return files

    except Exception as e:
        print(f"Error: {e}")
        return []


def get_folder_names(directory_path, sort=True, exclude_hidden=False):
    """
    Get folder names from a directory with options.

    Args:
        directory_path (str): Path to the directory
        sort (bool): Whether to sort folder names alphabetically
        exclude_hidden (bool): Whether to exclude hidden folders (starting with .)

    Returns:
        list: List of folder names
    """
    try:
        # Convert to Path object for better handling
        path = Path(directory_path)

        # Check if path exists
        if not path.exists():
            raise FileNotFoundError(f"Directory '{directory_path}' not found.")

        # Check if it's a directory
        if not path.is_dir():
            raise NotADirectoryError(f"'{directory_path}' is not a directory.")

        # Get all directories
        folders = []
        for item in path.iterdir():
            if item.is_dir():
                folder_name = item.name
                # Skip hidden folders if requested
                if exclude_hidden and folder_name.startswith('.'):
                    continue
                folders.append(folder_name)

        # Sort if requested
        if sort:
            folders.sort()

        return folders

    except Exception as e:
        print(f"Error: {e}")
        return []
print('Importing modules...')
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tqdm.auto import tqdm
from transformers import OlmoHybridModel, OlmoHybridConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import os
import random
import pickle
from typing import Tuple

UNKNOWN_TOKEN = "<UNK>"
PADDING_TOKEN = "<PAD>"
BEGIN_OF_STREAM_TOKEN = "<BOS>"
END_OF_STREAM_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PADDING_TOKEN, BEGIN_OF_STREAM_TOKEN, END_OF_STREAM_TOKEN, UNKNOWN_TOKEN]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
MODEL_NAME = 'UNAMED_MODEL'


def set_seed(seed):
    """Set seeds for reproducibility across different libraries."""
    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # NumPy seed
    np.random.seed(seed)
    
    # Python random module seed
    random.seed(seed)

def worker_init_fn(worker_id):
    """
    Picklable function to set seeds for each worker.
    Use a global seed or pass the seed as an argument when creating the DataLoader.
    """
    # Use a global seed or a seed passed through a global variable
    set_seed(457 + worker_id)

class TextDataset(Dataset):
    
    def __init__(self, texts, seq_len: int, tokenizer, savethis:bool=False, loading_prev:bool=False, prior_name:str = "unnamed_model", moddable_suffix:str='training'):

        self.seq_len = seq_len
            
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id(PADDING_TOKEN)
        
        self.sequences = texts
        self.loading_previous = loading_prev
        self.previous_model_name = prior_name
        self.save_this = savethis
        
        if texts == "Fake":
            return
            
        if not loading_prev:
            print("Splitting into chunks")
            #chunks = [chunk.strip() for chunk in texts.split(SEP_TOKEN) if len(chunk.strip()) > 0]
            print(f"We have {len(texts)} chunks to process")
            size_of_base_text = 0
            
            if self.save_this:
                print(f"Saving encoded corpus for later use...")
                _flat    = np.array([token for document in self.sequences for token in document], dtype=np.uint16)
                _lengths = np.array([len(document) for document in self.sequences],  dtype=np.uint32)
                np.save(f'models/{MODEL_NAME}/{MODEL_NAME}_{moddable_suffix}_flat.npy', _flat)
                np.save(f'models/{MODEL_NAME}/{MODEL_NAME}_{moddable_suffix}_lengths.npy', _lengths)
                del _flat, _lengths
        
        if loading_prev:
            print("Loading an encoded corpus...")
            #with open(f'models/{prior_name}/{prior_name}_{moddable_suffix}_data.pkl', 'rb') as file:
            #   chunk_token_ids = pickle.load(file)
            #chunk_token_ids = np.load(f'models/{prior_name}/{prior_name}_{moddable_suffix}_data.npy').tolist()
            _flat    = np.load(f'models/{prior_name}/{prior_name}_{moddable_suffix}_flat.npy')
            _lengths = np.load(f'models/{prior_name}/{prior_name}_{moddable_suffix}_lengths.npy')
            # np.split on cumulative offsets reconstructs the ragged list in one call
            self.sequences = np.split(_flat, np.cumsum(_lengths[:-1]))
            del _flat, _lengths
                    
        print('Converting sequences from python-list to numpy')
        self.sequences = np.array(self.sequences, dtype=np.uint16)
        
        
        print(f"Created {len(self.sequences):,} training sequences")
        print(f"We will be training on {len(self.sequences) * self.seq_len} tokens.")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        #chunk = self.sequences[idx]
        chunk = self.sequences[idx].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        #x = torch.tensor(chunk[:-1], dtype=torch.long)
        #y = torch.tensor(chunk[1:], dtype=torch.long)
        mask = (x != self.pad_id).long()
        
        return x, y, mask
    
    def split_array_randomly(self, split_percentage, seed=None):
        """Split dataset into train/val."""
        if seed is not None:
            random.seed(seed)
        
        split_percentage = max(0, min(1, split_percentage))
        sequences_copy = self.sequences.copy()
        random.shuffle(sequences_copy)
        
        num_first_split = int(len(sequences_copy) * split_percentage)
        first_split = sequences_copy[:num_first_split]
        second_split = sequences_copy[num_first_split:]
        
        return first_split, second_split


class OptimizedTrainer:
    
    def __init__(self, model: OlmoHybridModel, tokenizer: Tokenizer, config: OlmoHybridConfig, adamw_beta_1, adamw_beta2, label_smooth, pref_suf='',
                 learning_rate: float = 3e-4, weight_decay: float = 0.1, device: str = 'cuda',
                 gradient_checkpointing: bool = False, compile_model: bool = True):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.max_lr = learning_rate
        self.weight_decay_var = weight_decay
        self.label_smoothing = label_smooth
        self.additional_pref_suf = pref_suf
        
        
        # Gradient checkpointing (optional - trades compute for memory)
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Compile model for faster execution (PyTorch 2.0+)
        if device != 'cpu' and compile_model and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        
        # Enable memory efficient attention
        if hasattr(F, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        
        # Use AdamW with fused implementation (faster)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(adamw_beta_1, adamw_beta2),  # From optimized code
            eps=1e-8,
            fused=True if device == 'cuda' else False
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id(PADDING_TOKEN), label_smoothing=self.label_smoothing)
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')
        self.use_amp = device == 'cuda'
    
    # CREDIT TO DEEPSEEK FOR GRADIENT CHECKPOINTING
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        
        # Handle compiled model
        if hasattr(self.model, '_orig_mod'):
            model = self.model._orig_mod
        else:
            model = self.model
        
        # Check if the model has a built-in gradient checkpointing method
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Enabled built-in gradient checkpointing")
            return
        
        # If the model uses HuggingFace-style layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            for layer in layers:
                layer.gradient_checkpointing = True
            print(f"Enabled gradient checkpointing for {len(layers)} layers")
            return
        
        # Fallback: try to find layers recursively
        for name, module in model.named_modules():
            if 'block' in name or 'layer' in name:
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(
                            original_forward, *args, use_reentrant=False, **kwargs
                        )
                    
                    module.forward = checkpointed_forward
        
        print("Applied gradient checkpointing to identified layers")
    
    def train(self, training_start_time, train_data, epochs: int, batch_size: int,
              eval_interval: int = 600, save_interval: int = 2000, 
              gradient_accumulation_steps: int = 4, current_epoch: int = 0, clean_slate_save: bool = True, loading_prev:bool=False, prior_name:str = "unnamed_model"):
        """Train with all optimizations enabled."""
        
        config_info = f"{MODEL_NAME} Configuration = " + "{" +f"""
\t"Suffix" = {self.additional_pref_suf}
\t"Hidden Layers" = {self.config.num_hidden_layers},
\t"Attention Heads" = {self.config.num_attention_heads},
\t"Key Heads" = {self.config.num_key_value_heads},
\t"Embedding Dim" = {self.config.hidden_size},
\t"MLP Dim" = {self.config.intermediate_size},
\t"Vocab Size" = {self.config.vocab_size},
\t"Max Seq Len" = {self.config.max_position_embeddings},
\t"Weight Init Range = {self.config.initializer_range},
\t"Attention Dropout = {self.config.attention_dropout},
\t"Tie Embeddings = {self.config.tie_word_embeddings},
\t"Label Smoothing = {self.label_smoothing},
\t"Max LR" = {self.max_lr},
\t"Batch Size" = {batch_size},
\t"Gradient Accumulation Steps" = {gradient_accumulation_steps},
\t"Eval Interval" = {eval_interval},
\t"Save Interval" = {save_interval},
\t"Max Epochs" = {epochs},
\t"Training Start Time" = {training_start_time}
""" + "}"

        
        os.makedirs(os.path.dirname(f"models/{MODEL_NAME}/"), exist_ok=True)
        with open(f'models/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_configuration.txt', 'w') as file:
            file.write(config_info)
        
        if clean_slate_save:    
            self.save_checkpoint(
                self.model,
                f'{MODEL_NAME}', 
                f'{self.additional_pref_suf}_CleanSlate', 
                self.tokenizer
            )
            
        # Create datasets
        dataset = TextDataset(train_data, self.config.max_position_embeddings, self.tokenizer, clean_slate_save, loading_prev, prior_name, 'training')
        training_dataset = TextDataset("Fake", self.config.max_position_embeddings, self.tokenizer)
        testing_dataset = TextDataset("Fake", self.config.max_position_embeddings, self.tokenizer)
        
        training_dataset.sequences, testing_dataset.sequences = dataset.split_array_randomly(0.95, 42)
        del dataset
        
        print(f"Training on: {len(training_dataset.sequences):,} sequences")
        print(f"Validating on: {len(testing_dataset.sequences):,} sequences")
        
        # Create dataloaders with optimized settings
        dataloader = DataLoader(
            training_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,  # Parallel data loading
            pin_memory=False,  # Faster data transfer to GPU
            persistent_workers=True,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn
        )
        test_b_size = batch_size
        if len(testing_dataset.sequences) < batch_size:
            test_b_size /= 8
        test_b_size = int(test_b_size)
        testing_dataloader = DataLoader(
            testing_dataset, 
            batch_size=test_b_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn
        )
        
        # Cosine annealing with warmup
        warmup_steps = (len(dataloader) / gradient_accumulation_steps) // 5
        total_steps = (len(dataloader) // gradient_accumulation_steps) * epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)\
        
        # Training loop
        self.model.train()
        print(f"Initialization took {(time.time() - training_start_time):.2f}s")
        
        global_step = 0
        losses = []
        all_training_loss = []
        
        validation_losses = []
        testing_losses = []
        best_val_loss = float('inf')
        best_test_loss = float('inf')
        times_test_loss_has_worsened = 0
        total_times_test_loss_has_worsened = 0
        
        last_save_time = time.time()
        last_eval_time = last_save_time
        mandated_end_to_training = False
        t_colour_str = '\x1B[38;5;229m'
        for epoch in range(epochs-current_epoch):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1+current_epoch}/{epochs}")
            epoch_losses = []
            accumulated_loss = 0.0
            
            for batch_idx, (data_input, data_output, mask) in enumerate(pbar):
                data_input = data_input.to(self.device, non_blocking=True)
                data_output = data_output.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                
                
                loss = 0.0
                # Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(data_input, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), data_output.view(-1))
                    
                    # Scale loss for gradient accumulation
                loss /= gradient_accumulation_steps
                accumulated_loss += loss
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if self.use_amp:
                        # Unscale and clip gradients
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    #scheduler.step(metrics=accumulated_loss)
                    
                    # Logging
                    epoch_losses.append(accumulated_loss.item())
                    all_training_loss.append(accumulated_loss.item())
                    
                    pbar.set_postfix({
                        'loss': f'{accumulated_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    accumulated_loss = 0.0
                    
                    global_step += 1
                    
                    # Evaluation
                    if global_step % eval_interval == 0 or time.time() - last_eval_time >= 1800:
                        #val_loss = self._calculate_validation_loss(validation_dataloader)
                        test_loss = self._calculate_validation_loss(testing_dataloader)
                        #validation_losses.append((global_step, val_loss))
                        testing_losses.append((global_step, test_loss))
                        validation_losses.append((global_step, test_loss))
                        os.makedirs(os.path.dirname(f'tracking/{MODEL_NAME}/'), exist_ok=True)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_avg_tr_loss.pkl', 'wb') as file:
                            pickle.dump(losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_tr_loss.pkl', 'wb') as file:
                            pickle.dump(all_training_loss, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_val_loss.pkl', 'wb') as file:
                            pickle.dump(validation_losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_test_loss.pkl', 'wb') as file:
                            pickle.dump(testing_losses, file)
                        
                        # Save best model
                        t_colour_str = '\x1B[38;5;229m'
                        
                        if (test_loss < best_test_loss):
                            divi = min(10.0, best_test_loss) / max(test_loss, 0.001)
                            
                            if divi > 1.03:
                                t_colour_str = '\x1B[38;5;46m'
                            elif divi > 1.01:
                                t_colour_str = '\x1B[38;5;40m'
                            elif divi > 1.005:
                                t_colour_str = '\x1B[38;5;2m'
                            elif divi > 1.001:
                                t_colour_str = '\x1B[38;5;10m'
                            
                            best_test_loss = test_loss
                            times_test_loss_has_worsened = 0
                            self.save_checkpoint(self.model, f'{MODEL_NAME}', f'{self.additional_pref_suf}_best_test', self.tokenizer)
                        else:
                            times_test_loss_has_worsened += 1
                            total_times_test_loss_has_worsened += 1
                            if (test_loss > best_test_loss * 1.03):
                                t_colour_str = '\x1B[38;5;196m'
                                print("\x1B[38;5;196m\t[WARNING]: TEST LOSS EXCEEDS BEST BY 3%!!!\x1B[38;5;252m")
                                mandated_end_to_training = True
                            elif times_test_loss_has_worsened >= 5:
                                t_colour_str = '\x1B[38;5;196m'
                                print("\x1B[38;5;196m\t[WARNING]: TEST LOSS WORSENED 5x IN A ROW!!!\x1B[38;5;252m")
                                mandated_end_to_training = True
                            elif total_times_test_loss_has_worsened >= 8 and times_test_loss_has_worsened >= 2:
                                t_colour_str = '\x1B[38;5;196m'
                                print("\x1B[38;5;196m\t[WARNING]: TEST LOSS HAS PLATEAUED!!!\x1B[38;5;252m")
                                mandated_end_to_training = True
                            else:
                                if times_test_loss_has_worsened > 4:
                                    t_colour_str = '\x1B[38;5;160m'
                                elif times_test_loss_has_worsened > 3:
                                    t_colour_str = '\x1B[38;5;161m'
                                elif times_test_loss_has_worsened > 2:
                                    t_colour_str = '\x1B[38;5;203m'
                                elif times_test_loss_has_worsened > 1:
                                    t_colour_str = '\x1B[38;5;167m'
                                elif times_test_loss_has_worsened > 0:
                                    t_colour_str = '\x1B[38;5;209m'
                                
                                
                        # print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}, Test Loss: {t_colour_str}{test_loss:.4f}\x1B[38;5;252m")
                        print(f"\nv Step {global_step} - Test Loss: {t_colour_str}{test_loss:.4f}\x1B[38;5;252m")
                        
                        # Generate samples
                        self._generate_sample()
                        
                        print(f"\n^ Step {global_step} - Test Loss: {t_colour_str}{test_loss:.4f}\x1B[38;5;252m")
                        
                        last_eval_time = time.time()
                    
                    # Periodic checkpoint
                    if global_step % save_interval == 0 or time.time() - last_save_time >= 1800 or mandated_end_to_training:
                        os.makedirs(os.path.dirname(f'tracking/{MODEL_NAME}/'), exist_ok=True)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_avg_tr_loss.pkl', 'wb') as file:
                            pickle.dump(losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_tr_loss.pkl', 'wb') as file:
                            pickle.dump(all_training_loss, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_val_loss.pkl', 'wb') as file:
                            pickle.dump(validation_losses, file)
                        with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_test_loss.pkl', 'wb') as file:
                            pickle.dump(testing_losses, file)
                        # print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
                        print(f"\n^ Step {global_step} - Test Loss: {t_colour_str}{test_loss:.4f}\x1B[38;5;252m")
                        self.model.save_checkpoint(
                            f'{MODEL_NAME}', 
                            f'{self.config.additional_pref_suf}_step_{global_step}', 
                            self.tokenizer
                        )
                        last_save_time = time.time()
                if mandated_end_to_training:
                    break
            if mandated_end_to_training:
                break
            
            # End of epoch
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            #val_loss = self._calculate_validation_loss(validation_dataloader)
            test_loss = self._calculate_validation_loss(testing_dataloader)
            #validation_losses.append((global_step, val_loss))
            testing_losses.append((global_step, test_loss))
            validation_losses.append((global_step, test_loss))
            
            t_colour_str = '\x1B[38;5;229m'
            #if (val_loss < best_val_loss):
            #    best_val_loss = val_loss
            if (test_loss < best_test_loss):
                divi = min(10.0, best_test_loss) / max(test_loss, 0.001)
                
                if divi > 1.03:
                    t_colour_str = '\x1B[38;5;46m'
                elif divi > 1.01:
                    t_colour_str = '\x1B[38;5;40m'
                elif divi > 1.005:
                    t_colour_str = '\x1B[38;5;2m'
                elif divi > 1.001:
                    t_colour_str = '\x1B[38;5;10m'
                
                best_test_loss = test_loss
                times_test_loss_has_worsened = 0
                self.save_checkpoint(self.model, f'{MODEL_NAME}', f'{self.additional_pref_suf}_best_test', self.tokenizer)
            else:
                times_test_loss_has_worsened += 1
                total_times_test_loss_has_worsened += 1
                if (test_loss > best_test_loss * 1.03):
                    t_colour_str = '\x1B[38;5;83m'
                    print("\x1B[38;5;196m\t[WARNING]: TEST LOSS EXCEEDS BEST BY 3%!!!\x1B[38;5;252m")
                    mandated_end_to_training = True
                elif times_test_loss_has_worsened >= 5:
                    t_colour_str = '\x1B[38;5;196m'
                    print("\x1B[38;5;196m\t[WARNING]: TEST LOSS WORSENED 5x IN A ROW!!!\x1B[38;5;252m")
                    mandated_end_to_training = True
                elif total_times_test_loss_has_worsened >= 8 and times_test_loss_has_worsened >= 2:
                    t_colour_str = '\x1B[38;5;196m'
                    print("\x1B[38;5;196m\t[WARNING]: TEST LOSS HAS PLATEAUED!!!\x1B[38;5;252m")
                    mandated_end_to_training = True
                else:
                    if times_test_loss_has_worsened > 4:
                        t_colour_str = '\x1B[38;5;160m'
                    elif times_test_loss_has_worsened > 3:
                        t_colour_str = '\x1B[38;5;161m'
                    elif times_test_loss_has_worsened > 2:
                        t_colour_str = '\x1B[38;5;203m'
                    elif times_test_loss_has_worsened > 1:
                        t_colour_str = '\x1B[38;5;167m'
                    elif times_test_loss_has_worsened > 0:
                        t_colour_str = '\x1B[38;5;209m'
            
            os.makedirs(os.path.dirname(f'tracking/{MODEL_NAME}/'), exist_ok=True)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_avg_tr_loss.pkl', 'wb') as file:
                pickle.dump(losses, file)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_tr_loss.pkl', 'wb') as file:
                pickle.dump(all_training_loss, file)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_val_loss.pkl', 'wb') as file:
                pickle.dump(validation_losses, file)
            with open(f'tracking/{MODEL_NAME}/{MODEL_NAME}{self.additional_pref_suf}_step_{global_step}_test_loss.pkl', 'wb') as file:
                pickle.dump(testing_losses, file)
            
            # print(f'\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {t_colour_str}{test_loss:.4f}')
            print(f'\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}, Test Loss: {t_colour_str}{test_loss:.4f}\x1B[38;5;252m')
            
            if not mandated_end_to_training:
                self._generate_sample()
        
            if mandated_end_to_training:
                break
        
        if mandated_end_to_training:
            print("\x1B[38;5;196m\t[WARNING]: ENDING TRAINING EARLY!!!\x1B[38;5;214m")
            print(f"\x1B[38;5;49m\tFinished with Best Test Loss of: \x1B[38;5;46m{best_test_loss}\x1B[38;5;214m")
        
        return losses, validation_losses, testing_losses
    
    def save_checkpoint(self, model:OlmoHybridModel, path: str, path_suffix: str, tokenizer: Tokenizer):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(f"models/{path}/"), exist_ok=True)
        torch.save({
            'config': self.config,
            'model_state_dict': model.state_dict()
        }, f"models/{path}/{path+path_suffix}"+'.pth')
        tokenizer.save(f"models/{path}/{path}_tokenizer{path_suffix}.json")
        print(f'Checkpoint saved to models/{path}')
    
    @staticmethod
    def load_checkpoint(path: str, path_suffix: str, device: str = 'cpu') -> Tuple[OlmoHybridModel, Tokenizer]:
        """Load model from checkpoint."""
        with torch.serialization.safe_globals([OlmoHybridConfig]):
            checkpoint = torch.load(f"models/{path}/{path+path_suffix}"+'.pth', map_location=device, weights_only=False)
        model = OlmoHybridModel(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = Tokenizer.from_file(f"models/{path}/{path}_tokenizer{path_suffix}.json")
        
        return model.to(device), tokenizer
    
    def _calculate_validation_loss(self, validation_dataloader):
        """Calculate validation loss efficiently."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data_input, data_output, mask in validation_dataloader:
                data_input = data_input.to(self.device, non_blocking=True)
                data_output = data_output.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(data_input, mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), data_output.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 1000:
                    break
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def _generate_sample(self):
        """Generate samples during training."""
        self.model.eval()
        
        class_prompts = [
            "<BOS><english><RESP>",
            "<BOS><french><RESP>",
            "<BOS><russian><RESP>",
            "<BOS><spanish><RESP>",
            "<BOS><chinese><RESP>",
            "<BOS><muslim><RESP>",
            "<BOS><jewish><RESP>",
            "<BOS><modern-egyptian><RESP>"
        ]
        
        name_prompts = [
            "<BOS>Christopher Green<RESP>", # english
            "<BOS>Jessica Morris<RESP>", # english
            "<BOS>Émile Guillaume<RESP>", # french
            "<BOS>Félicité Cazenave<RESP>", # french
            "<BOS>Shustelyov Maksim Igorevich<RESP>", # russian
            "<BOS>Zhukova Agnessa Nikitovna<RESP>", # russian
            "<BOS>Jose Angel Zanhuesa<RESP>", # spanish
            "<BOS>Carlota España<RESP>", # spanish
            "<BOS>Teng Zhelan<RESP>", # chinese
            "<BOS>Hao Zexi<RESP>", # chinese
            "<BOS>Muhyddeen al-Zaher<RESP>", # muslim
            "<BOS>Sharaf al-Firman<RESP>", # muslim
            "<BOS>Don Mendenhall<RESP>", # jewish
            "<BOS>Neora Singer<RESP>", # jewish
            "<BOS>Anwar Malouf<RESP>", # modern-egyptian
            "<BOS>Kafele Ghanem<RESP>" # modern-egyptian
        ]
        
        
        for prompt in class_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                generate_ids = model.generate(inputs.input_ids, max_length=self.config.max_position_embeddings)
                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print(f"PROMPT: {prompt}")
                print(f"GENERATED: {output}")
                
            except Exception as e:
                print(f'Error generating for prompt "{prompt}": {e}')
        
        for prompt in name_prompts:
            encoded = self.tokenizer.encode(prompt)
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, 5)
                print(f"Top-{top_k} next-token predictions for: {prompt!r}\n")
                for token_id, p in zip(top_indices.cpu(), top_probs.cpu()):
                    print(f"{tokenizer.decode(token_id.item()):15s}  {p*100:6.2f}%")
            except Exception as e:
                print(f'Error generating for prompt "{prompt}": {e}')
        
        
        self.model.train()



def train_tokenizer(vocab_size, special_tokens_param, unk_tok, added_toks, training_data, train=True):
        """Train BPE tokenizer."""
        bpe_trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens_param)
        tokenizer = Tokenizer(models.BPE(unk_token=unk_tok))
        tokenizer.add_tokens(class_tokens)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()
        
        if train:
            tokenizer.train_from_iterator(training_data, trainer=bpe_trainer)
        
        return tokenizer

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total:,}')
    return total



if __name__ == "__main__":
    VOCAB_SIZE = 8192
    HIDDEN_LAYERS = 4
    NUM_ATN_HEADS = 4
    NUM_KEY_HEADS = 4
    EMBED_DIM = 128
    MLP_SIZE = 512
    WEIGHT_INIT_RANGE = 0.02 # DEFAULT
    TIE_EMBED = True
    SEQ_LEN = 32
    BATCH_SIZE = 8192
    GRADIENT_ACCUM_STEPS = 4
    MAX_LEARN_RATE = 3e-4
    NUM_EPOCHS = 5
    SAVE_STEPS = 200
    EVAL_STEPS = 200
    
    OPTIMIZER_BETA1 = 0.95
    OPTIMIZER_BETA2 = 0.995
    LABEL_SMOOTHING = 0.0001
    
    no_error = False
    
    config = OlmoHybridConfig()
    
    
    while not no_error:
        try:
            print("\x1B[38;5;123mMODEL INITIALIZATION:\x1B[38;5;252m")       
            MODEL_NAME = input("\x1B[38;5;252m\tModel Name: \x1B[38;5;214m")
            HIDDEN_LAYERS = int(input("\x1B[38;5;252m\tLayers: \x1B[38;5;214m"))  
            NUM_ATN_HEADS = int(input("\x1B[38;5;252m\tAttention-Heads: \x1B[38;5;214m"))
            NUM_KEY_HEADS = int(input("\x1B[38;5;252m\tKey-Heads: \x1B[38;5;214m"))
            EMBED_DIM = int(input("\x1B[38;5;252m\tEmbedding Dimension: \x1B[38;5;214m"))
            MLP_SIZE = int(input("\x1B[38;5;252m\tIntermediate Size: \x1B[38;5;214m"))
            WEIGHT_INIT_RANGE = float(input("\x1B[38;5;252m\tWeight Init Range: \x1B[38;5;214m"))
        
            print("\x1B[38;5;123mDATA INITIALIZATION:\x1B[38;5;252m")
            VOCAB_SIZE = int(input("\x1B[38;5;252m\tVocab Size: \x1B[38;5;214m"))
            SEQ_LEN = int(input("\x1B[38;5;252m\tMaximum Sequence Length: \x1B[38;5;214m"))
        
            print("\x1B[38;5;123mTRAINING INITIALIZATION:\x1B[38;5;252m")
            BATCH_SIZE = int(input("\x1B[38;5;252m\tBatch Size: \x1B[38;5;214m"))
            GRADIENT_ACCUM_STEPS = int(input("\x1B[38;5;252m\tGradient Accumulation Steps: \x1B[38;5;214m"))
            MAX_LR = float(input("\x1B[38;5;252m\tMaximum Learning Rate: \x1B[38;5;214m"))
            NUM_EPOCHS = int(input("\x1B[38;5;252m\tNumber of Epochs: \x1B[38;5;214m"))
            
            print("\x1B[38;5;123mLOGGING INITIALIZATION:\x1B[38;5;252m")
            EVAL_STEPS = int(input("\x1B[38;5;252m\tEvaluate Every X Optimizer Steps: \x1B[38;5;214m"))
            SAVE_STEPS = int(input("\x1B[38;5;252m\tSave Every X Optimizer Steps: \x1B[38;5;214m"))
            
            no_error = True
        except KeyboardInterrupt:
            print("\x1B[38;5;252mClosing...")
            raise Exception('Closing')
        except:
            print("\x1B[38;5;252mERROR. You entered something incorrectly. Restarting.")
    init_start_time = time.time()
    print(f"\x1B[38;5;123mBEGINING TRAINING FOR: {MODEL_NAME}\x1B[38;5;252m")
    
    config.num_attention_heads = NUM_ATN_HEADS
    config.num_key_value_heads = NUM_KEY_HEADS
    config.vocab_size = VOCAB_SIZE
    config.hidden_size = EMBED_DIM
    config.intermediate_size = MLP_SIZE
    config.num_hidden_layers = HIDDEN_LAYERS
    config.initializer_range = WEIGHT_INIT_RANGE
    config.max_position_embeddings = SEQ_LEN
    config.tie_word_embeddings = TIE_EMBED
    config.attention_dropout = 0.25
        

    folder_names = get_folder_names('data')

    class_tokens = ['<RESP>']

    corpus_data = []
    for folder in tqdm(folder_names, desc='Processing folders...'):
        files_in_folder = get_file_names(f'data/{folder}')
        class_tokens.append(f'<{folder}>')
        for file in tqdm(files_in_folder, desc=f'Processing {folder}...'):
            ac_identifier = file.split("-names.txt")[0]
            class_tokens.append(f'<{ac_identifier}>')
            with open(f'data/{folder}/{file}', 'r', encoding='utf-8') as r_file:
                f_data = r_file.read()
                f_data_by_line = f_data.split("\n")
                
                corpus_data.append(f_data_by_line)

   

    
    TOKENIZER = train_tokenizer(VOCAB_SIZE, SPECIAL_TOKENS, UNKNOWN_TOKEN, class_tokens, corpus_data)
    TOKENIZER.padding_side = "left"
    
    #print(tokenizer.get_vocab())


    build_actual_training_data = []

    bigg_len = 0
    resp_tok = TOKENIZER.token_to_id(class_tokens[0])
    bos_tok = TOKENIZER.token_to_id(BEGIN_OF_STREAM_TOKEN)
    eos_tok = TOKENIZER.token_to_id(END_OF_STREAM_TOKEN)
    config.pad_token_id = TOKENIZER.token_to_id(PADDING_TOKEN)
    config.bos_token_id = bos_tok
    config.eos_token_id = eos_tok
    
    
    
    for folder in tqdm(folder_names, desc='Processing folders...'):
        files_in_folder = get_file_names(f'data/{folder}')
        class_tokens.append(f'<{folder}>')
        for file in tqdm(files_in_folder, desc=f'Processing {folder}...'):
            ac_identifier = file.split("-names.txt")[0]
            class_tok = TOKENIZER.token_to_id(f'<{ac_identifier}>')
            with open(f'data/{folder}/{file}', 'r', encoding='utf-8') as r_file:
                f_data = r_file.read()
                f_data_by_line = f_data.split("\n")
                
                for name in f_data_by_line:
                    token_ids_class_first = [bos_tok, class_tok, resp_tok]
                    enc_name = TOKENIZER.encode(name).ids
                    token_ids_class_first.extend(enc_name)
                    token_ids_class_first.append(eos_tok)
                    bigg_len = max(len(token_ids_class_first), bigg_len)
                    token_ids_class_first.extend([config.pad_token_id]*(SEQ_LEN-len(token_ids_class_first)))
                    
                    
                    build_actual_training_data.append(token_ids_class_first)
                    
                    token_ids_entry_first = [bos_tok]
                    token_ids_entry_first.extend(enc_name)
                    token_ids_entry_first.extend([resp_tok, class_tok, eos_tok])
                    bigg_len = max(len(token_ids_entry_first), bigg_len)
                    token_ids_entry_first.extend([config.pad_token_id]*(SEQ_LEN - len(token_ids_entry_first)))
                    build_actual_training_data.append(token_ids_entry_first)
                    
                    
                    # In the format of:
                    # <BOS><CLASS_TOKEN><RESP><NAME><EOS>
                    # <BOS><NAME><RESP><CLASS_TOKEN><EOS>
    print(f"MAXIMUM SEQUENCE LENGTH: {bigg_len}")
    
    print("Creating model...")
    MODEL = OlmoHybridModel(config)
    
    count_parameters(MODEL)
    
    trainer = OptimizedTrainer(
        MODEL,
        TOKENIZER,
        config,
        OPTIMIZER_BETA1,
        OPTIMIZER_BETA2,
        LABEL_SMOOTHING,
        '',
        device=device,
        learning_rate=MAX_LEARN_RATE,
        weight_decay=0.3,
        gradient_checkpointing=True,
        compile_model=False
    )   
    
    print("Starting training...")
    print(len(build_actual_training_data))
    losses, validation_losses, testing_losses = trainer.train(
        init_start_time,
        build_actual_training_data,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        eval_interval=EVAL_STEPS,
        save_interval=SAVE_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        clean_slate_save=True,
        loading_prev=False,
        prior_name="unnamed_model",
    )
    
    
    
    
                
    #n_str = f"<BOS><{ac_identifier}><RESP>" + f"<EOS><BOS><{ac_identifier}><RESP>".join(f_data_by_line)
    #n_str2 = "".join([f"<BOS>{entry}<RESP><{ac_identifier}><EOS>" for entry in f_data_by_line]) # CREDIT TO DEEPSEEK: 'Use list comprehension with join instead of loop'
    
    #token_ids_entry_first = [token for entry in f_data_by_line for token in tokenizer.encode(f"<BOS>{entry}<RESP><{ac_identifier}><EOS>")] # CREDIT TO GPT-5 mini
    #token_ids_class_first = [token for entry in f_data_by_line for token in tokenizer.encode(f"<BOS>{ac_identifier}<RESP><{entry}><EOS>")] # CREDIT TO GPT-5 mini
    
    