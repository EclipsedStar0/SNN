import pickle
with open('tracking/optim_dump_test_step_600_tr_loss.pkl', 'rb') as file:
    losses = pickle.load(file)
    print(losses)