from models import *
from model_utils import *
import numpy as np

class parallel(nn.Module):

    def __init__(self,Encoder,  mem_dim=149, shrink_thres=0.0025):
        super(parallel, self).__init__()


        self.AE_encoder = nn.Sequential(
            nn.Linear(5, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 3),
            torch.nn.ReLU()

        )

        # self.AE_decoder = nn.Sequential(
        #     nn.Linear(5, 5)
        #
        # )

        self.MemAE_encoder = nn.Sequential(

            nn.Linear(5, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 3),
            torch.nn.ReLU()

        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=3, shrink_thres=shrink_thres)
        # self.MemAE_decoder = nn.Sequential(
        #     nn.Linear(5, 5)
        #
        # )

        self.VAE_Encoder = Encoder
        # self.VAE_Decoder = Decoder

        self.g_decoder=  nn.Sequential(

            nn.Linear(9, 7),
            torch.nn.ReLU(),
            nn.Linear(7, 5)

        )



    def reparameterization(self, mean, var):
            epsilon = torch.randn_like(var)  # sampling epsilon
            z = mean + var * epsilon  # reparameterization trick
            return z

    def forward(self, x):
        f1 = self.AE_encoder(x)

        f2 = self.MemAE_encoder(x)
        res_mem = self.mem_rep (f2)
        f2 = res_mem['output']
        att_MemAE = res_mem['att']


        mean, log_var = self.VAE_Encoder(x)
        f3 = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)


        final_output= self.g_decoder(torch.cat((f1,f2,f3),1))

        return {'output': final_output, 'att': att_MemAE}

