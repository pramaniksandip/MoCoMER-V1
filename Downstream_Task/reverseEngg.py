import torch
ckpt = torch.load('/workspace/nemo/data/RESEARCH/HMER_CoMER/CoMER/epoch=143-step=216143-val_ExpRate=0.6274.ckpt', map_location='cpu')
print(ckpt.keys())  # Look for 'hparams', 'state_dict', etc.

print("Epoch:", ckpt['epoch'])
print("Global step:", ckpt['global_step'])
print("Hyperparameters:", ckpt['hyper_parameters'])
#print("Hyperparameters:", ckpt['state_dict'])
#print("state_dict:", ckpt['state_dict'])
#print("optimizer_states:", ckpt['optimizer_states'])
print("lr_schedulers:", ckpt['lr_schedulers'])
print("hparams_name:", ckpt['hparams_name'])


#print(ckpt['hparams'])