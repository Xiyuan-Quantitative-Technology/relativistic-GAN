# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:42:50 2021

@author: tanzheng
"""

import torch
import pickle
import numpy as np

from idseq_build import build_id_seq
from GAN_module import GAN
from GAN_trainer import GAN_Trainer

import time
with open('fullD1A1_pred_ECFP_input.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
    
input_smile, M_ECFP = data_list

all_smile = []
for i in input_smile:
    all_smile.append(str(i[0]))
# X_mol_idseq, X_char_vocab = build_id_seq(input_smile)
with open('D1A1_database_input.pkl', 'rb') as f:
    data_list = pickle.load(f)
    f.close()
    
    
tasks, x, test_smiles = data_list

for i in x:
    all_smile.append(str(i))
    
all_smile=np.array(all_smile)
X_mol_idseq, X_char_vocab = build_id_seq(all_smile)

#########################################################################
#set seed and device
seed=12
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Python random module.
torch.manual_seed(seed) 

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(36)

#########################################################################
# GAN parameters
emb_D = 64
encod_hidden_D = 256
decod_hidden_D = 512
lat_v_D = 128
bidir_flag = True
dis_hid_D = 512
gen_hid_D = 512

#training parameters
lr, batch_size = 0.0001, 256
# weight of recon loss
#recon_ratio = 1

#model generation
model = GAN(X_char_vocab, emb_D, encod_hidden_D, decod_hidden_D, \
            lat_v_D, bidir_flag, dis_hid_D, gen_hid_D)
trainer = GAN_Trainer(model, lr, batch_size)
N_epoch = 50

all_loss = []

all_model = []
# norm_ = np.ceil(len(input_smile) / batch_size)

for i in range(N_epoch):
    time_start = time.time()
    loss_train, each_model = trainer.train(X_mol_idseq, i)
    time_end = time.time()
    print(time_end  - time_start)
    all_model.append(each_model)
    print(i, float(loss_train['D_loss']), float(loss_train['G_loss']), float(loss_train['recon_loss']))
    
    all_loss.append(loss_train)
    
all_model_result = [all_model, all_loss]

with open('all_model_loss_0.0001 512 512 batch256 block1.pkl', 'wb') as f:
    pickle.dump(all_model_result, f)
    f.close()    
# GAN generation
generated_samples = model.samples_generation(1000)

# # GAN training
# N_epoch = 150
# all_loss = []

# for i in range(N_epoch):
#     time_start = time.time()
#     loss_train = trainer.train(X_mol_idseq, i)
#     time_end = time.time()
    
#     with open('loss_res.log', 'a+') as fw:
#         fw.writelines([str(i), ' '])
#         fw.writelines([loss_train['G_loss'], ' ', loss_train['D_loss'], ' ', loss_train['recon_loss'], '\n'])
#         fw.writelines([str(time_end-time_start), '\n'])
#         fw.close()
    
#     all_loss.append(loss_train)
    
# # GAN generation
# generated_samples = model.samples_generation(1000)

# # save to pickle
# all_model_result = [model, all_loss]

# with open('all_model_loss_0.001 128 512.pkl', 'wb') as f:
#     pickle.dump(all_model_result, f)
#     f.close()
