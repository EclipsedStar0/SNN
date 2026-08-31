import datasets
print("Loading...")
ds = datasets.load_dataset("storytracer/LoC-PD-Books") 
print("Saving...")
ds.save_to_disk("V:/CursorLocker/SNN/ANN/hf_dataset/LoC-PD-Books")
print("Done.")