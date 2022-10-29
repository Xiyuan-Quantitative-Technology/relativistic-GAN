# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:53:13 2021

@author: tanzheng
"""

import numpy as np
from itertools import chain
from define_vocabulary import CharVocab, char_set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GAN_Trainer():
    def __init__(self, model, lr, batch_size):
        self.model = model
        self.lr = lr
        self.bs = batch_size
        
        self.AE_optimizer = optim.Adam(chain(model.encoder.parameters(), model.decoder.parameters()), \
                                       lr=self.lr, betas=(0.9,0.99))
        
        self.G_optimizer = optim.Adam(model.generator.parameters(), \
                                      lr=self.lr, betas=(0.9,0.99))
        self.D_optimizer = optim.Adam(model.discriminator.parameters(), \
                                      lr=self.lr, betas=(0.9,0.99))
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def train(self, X_mol_idseq, epo_id):
        No_mol = len(X_mol_idseq)
        
        #epoch training
        loss_total = {'G_loss':0, 'D_loss':0, \
                      'discr_real_loss':0, 'discr_fake_loss':0, 'recon_loss':0}
            
        adversarial_loss = nn.BCEWithLogitsLoss()
        
        for i in range(0, No_mol, self.bs):
            #batch data prepare
            BX_mol_idseq = X_mol_idseq[i:i+self.bs]
            #BX_mol_idseq.to(self.device)
            
            temp_bs = len(BX_mol_idseq)         # temp batch size
            
            #ground truth
            real = torch.ones(temp_bs, 1)
            fake = torch.zeros(temp_bs, 1)
            
            # -----------------
            #  Train AE
            # -----------------
            
            latent_space = self.model.encoder_forward(BX_mol_idseq)
            recon_loss = self.model.decoder_forward(BX_mol_idseq, latent_space)
            
            self.AE_optimizer.zero_grad()
            recon_loss.backward()
            self.AE_optimizer.step()
            
            loss_total['recon_loss'] += recon_loss.item()
            
            ##########################################################
            if epo_id >= 5:
                # -----------------
                #  Train Generator
                # -----------------
                
                # Sample from noise
                ran_sample = self.model.random_sample_latent(temp_bs)
                # virtual latent space
                virtual_latent = self.model.generator_forward(ran_sample)
                
                fake_pred = self.model.discriminator_forward(virtual_latent)
                real_pred = self.model.discriminator_forward(latent_space.detach())
                
                G_loss = adversarial_loss(fake_pred - real_pred, real)
                
                self.G_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                discr_output = self.model.discriminator_forward(latent_space.detach())
                discr_fake_output = self.model.discriminator_forward(virtual_latent.detach())
                
                discr_real_loss = adversarial_loss(discr_output - discr_fake_output, real)
                discr_fake_loss = adversarial_loss(discr_fake_output - discr_output, fake)
                
                D_loss = 0.5 * (discr_real_loss + discr_fake_loss)
                
                self.D_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()
                
                
                loss_total['G_loss'] += G_loss.item()
                loss_total['D_loss'] += D_loss.item()
                loss_total['discr_real_loss'] += discr_real_loss.item()
                loss_total['discr_fake_loss'] += discr_fake_loss.item()
            
            else:
                loss_total['G_loss'] += 0
                loss_total['D_loss'] += 0
                loss_total['discr_real_loss'] += 0
                loss_total['discr_fake_loss'] += 0
                
            
        No_batch = np.ceil(No_mol/self.bs)
        for field, value in loss_total.items():
            loss_total[field] = str(value/No_batch)
        
        loss_mean = loss_total
        
        return loss_mean, self.model
            
        
        