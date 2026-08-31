import datasets
print("Loading...")
ds = datasets.load_from_disk("LoC-PD-Books")

tot_chara = 0
#for index in range(len(ds['train'])):
for index in range(2000):
    partial_sample = ds['train'][index]['text']
    tot_chara += len(partial_sample)
    partial_sample = None
print(tot_chara)
print(tot_chara * 0.25)
print(len(ds))
ds = None
