t_a = ''
t_b = ''
for index in range(328):
    t_a += 'a'
for index in range(1324):
    t_b += 'a'
t_a += 'c'*2048
t_b += 'c'*2048
wyrd = [[t_a, 2141709], [t_b, 530367]]

ctx_windows = [64, 128, 256, 512, 1024, 2048]

for padded_wrd, total_seq in wyrd:
    for window in ctx_windows:
        num_seq_without_padding = 0
        num_real_tok = 0
        num_partial_tok = 0
        num_padding = 0
        
        
        prst_index = 0
        ran_out_of_as = False
        while not ran_out_of_as:
            next_snippet = padded_wrd[prst_index:window+prst_index]
            num_real_tok += next_snippet.count('a')
            if next_snippet.count('c') < 1:
                num_seq_without_padding += 1
                prst_index += window // 2
            else:
                ran_out_of_as = True
                num_partial_tok = next_snippet.count('a')
                num_padding = next_snippet.count('c')
                break
        
        computation_waste = num_padding/(num_real_tok+num_padding)
        print(f'{window} tk ctx window | {num_seq_without_padding} full sequences of {window} tokens, 1 partial sequence of {num_partial_tok} tokens ({100*num_padding/window:.2f}%) padding; {100*computation_waste:.2f}% Computation Waste | {total_seq * (num_real_tok + num_padding):,} Trained Tks (includes padding) | {total_seq * num_real_tok:,} Trained Tks (not counting padding)')