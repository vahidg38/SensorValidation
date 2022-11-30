from model_utils import *
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import optuna
from optuna.trial import TrialState
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from integrated import *

#nn.Conv1d with a kernel size of 1 and nn.Linear give essentially the same results
class base_AE(nn.Module):
    def __init__(self):
        super(base_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(5, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 3),
            torch.nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 5)

        )


    def forward(self, x):
        f = self.encoder(x)
        output = self.decoder(f)
        return {'output': output, 'latent': f}

class base_AE_with_err (nn.Module):
    def __init__(self):
        super(base_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(6, 5),
            torch.nn.ReLU()

        )

        self.decoder = nn.Sequential(
            nn.Linear(5, 6)

        )


    def forward(self, x):
        f = self.encoder(x)
        output = self.decoder(f)
        return {'output': output, 'latent': f}

class build_base_AE(nn.Module):
    def __init__(self,trial):
        super(build_base_AE, self).__init__()
        n_layers = trial.suggest_int("n_layers", 1, 3)
        in_features = 5
        en_layers = []
        de_layers=[]
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 1, 5)
            en_layers.append(nn.Linear(in_features, out_features))
            de_layers.append(nn.Linear(out_features, in_features))
            en_layers.append(nn.ReLU())
            de_layers.append(nn.ReLU())


            in_features = out_features
        self.encoder= nn.Sequential(*en_layers)
        print(self.encoder)
        de_layers=de_layers[:-1] #remove the relu
        de_layers=de_layers[::-1] #reverse

        self.decoder=nn.Sequential(*de_layers)
        print(self.decoder)

    def forward(self, x):
            f = self.encoder(x)
            output = self.decoder(f)
            return {'output': output, 'latent': f}
class MemAE(nn.Module):
    def __init__(self, mem_dim=149, shrink_thres=0.0025):
        super(MemAE, self).__init__()

        self.encoder = nn.Sequential(

            nn.Linear(5, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 3),
            torch.nn.ReLU()

        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=3, shrink_thres=shrink_thres)
        self.decoder = nn.Sequential(
            nn.Linear(3, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 5)

        )

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att, 'latent': f}

class build_MemAE(nn.Module):
    def __init__(self, trial):

        super(build_MemAE, self).__init__()
        n_layers = trial.suggest_int("n_layers", 1, 3)
        in_features = 5
        en_layers = []
        de_layers = []
        out_features=2 #default value to be global
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 1, 5)
            en_layers.append(nn.Linear(in_features, out_features))
            de_layers.append(nn.Linear(out_features, in_features))
            en_layers.append(nn.ReLU())
            de_layers.append(nn.ReLU())
            in_features = out_features
        self.encoder = nn.Sequential(*en_layers)
        print(self.encoder)
        de_layers = de_layers[:-1]  # remove the relu
        de_layers = de_layers[::-1]  # reverse

        self.decoder = nn.Sequential(*de_layers)
        print(self.decoder)

        mem_dim=trial.suggest_int("n_units_l_{}".format(i), 50, 350)
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim= out_features, shrink_thres=0.0025)
        print(self.mem_rep)



    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att, 'latent': f}


class VAE(nn.Module):
    def __init__(self, Encoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 5)

        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)

        return {'output': x_hat, 'mean': mean, 'var': log_var, 'latent': z}


class MVAE(nn.Module):
    def __init__(self, Encoder, mem_dim=149,fe_dim=3, shrink_thres=0.0025):
        super(MVAE, self).__init__()
        self.Encoder = Encoder
        self.decoder = nn.Sequential  (#using memory autoencoder decoder
            nn.Linear(3, 4),
            torch.nn.ReLU(),
            nn.Linear(4, 5)

        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=fe_dim, shrink_thres=shrink_thres)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var

        res_mem = self.mem_rep(z)
        f = res_mem['output']
        att = res_mem['att']

        output = self.decoder(f)
        return {'output': output, 'att': att, 'latent': f}


def make_dir(model):
    if not os.path.exists(f'./{model}'):
        os.mkdir(f'./{model}')


class model():

    def __init__(self, args, train, train_n, test, test_faulty):
        self.args = args
        self.train = train
        self.train_n = train_n
        self.test = test
        self.test_faulty = test_faulty
        self.model_ = None
        self.loss_function = torch.nn.MSELoss()
        self.tr_entropy_loss_func = EntropyLossEncap()
        self.optimizer = None
        self.entropy_loss_weight = 0.0002
        self.mem = False
        self.entropy_loss_weight = 0
        self.pstart = args.pstart
        self.pstop = args.pstop

        print(f"{self.args.model} is selected")
        make_dir(self.args.model)

        match self.args.model:
            case "AE":
                self.model_ = base_AE()
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)
            case "DAE":
                self.model_ = base_AE()
                self.train_or=self.train
                self.train = self.train_n
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

            case "MAE":
                self.model_ = MemAE(mem_dim=self.args.memdim, shrink_thres=0.0025)
                self.mem = True
                self.entropy_loss_weight = 0.0002
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

            case "VAE":
                encoder = Encoder(input_dim=5, hidden_dim=4, latent_dim=3)
                self.model_ = VAE(Encoder=encoder)
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

            case "MVAE":
                encoder = Encoder(input_dim=5, hidden_dim=4, latent_dim=3)

                self.model_ = MVAE(Encoder=encoder, mem_dim=self.args.memdim, shrink_thres=0.0025)
                self.entropy_loss_weight = 0.0002
                self.mem = True
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)
            case "integrated":
                encoder = Encoder(input_dim=5, hidden_dim=4, latent_dim=3)
                self.train_or = self.train
                self.train= self.train_n
                self.model_ = parallel(mem_dim=self.args.memdim, shrink_thres=0.0025,Encoder=encoder)
                self.entropy_loss_weight = 0.0002
                self.mem = True
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

    def plot_memory(self, model, epoch):
        if not os.path.exists(f'./{self.args.model}/MemoryElements'):
            os.mkdir(f'./{self.args.model}/MemoryElements')
        plt.scatter(model.mem_rep.melements.detach().numpy()[:, 0], model.mem_rep.melements.detach().numpy()[:, 1])
        # plt.show()
        plt.ylabel("Feature2")
        plt.xlabel("Feature1")
        plt.title("Memory_Elements_MemAE")
        plt.savefig(f'./{self.args.model}/MemoryElements/Epoch_{epoch}.jpg', bbox_inches="tight", pad_inches=0.0)
        plt.clf()

    def train_model(self):
        writer = SummaryWriter(f'runs/{self.args.model}')

        wait = 0
        epoch_loss = []
        eot = False  # end of training
        for epoch in range(self.args.epochs):

            if self.mem:  # if model has memory module plot the elements during training
                #self.plot_memory(self.model_, epoch)
                pass

            if (eot == True):
                break
            losses = 0

            #print(f"epoch: {epoch}")

            iteration = len(self.train) // self.args.batch
            for i in range(iteration):
                # latent_mae=[]
                # outputs = []

                obs = torch.from_numpy(self.train.iloc[i * self.args.batch:(i + 1) * self.args.batch].to_numpy())

               # writer.add_graph(self.model_, obs.float(), use_strict_trace=False)
                reconstructed = self.model_(obs.float())
                #  print("obs")
                #  print(obs)
                # print("rec")
                # print(reconstructed['output'][0])
                if self.args.model == 'DAE' or self.args.model == 'integrated':
                    obs = torch.from_numpy(self.train_or.iloc[i * self.args.batch:(i + 1) * self.args.batch].to_numpy())
                # print(att_w)
                loss = self.loss_function(reconstructed['output'], obs.float())

                if self.mem:
                    att_w = reconstructed['att']
                    entropy_loss = self.tr_entropy_loss_func(att_w)
                    loss = loss + self.entropy_loss_weight * entropy_loss

                loss_val = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = losses + loss_val

            epoch_loss.append(losses / iteration)
            print(f"epoch_{epoch}_loss:  {losses / iteration}")
            writer.add_scalar("Loss/train", losses / iteration, epoch)

            if len(epoch_loss) > 2:
                if epoch_loss[epoch] > epoch_loss[epoch - 1]:
                    wait = wait + 1
                    if wait > self.args.patience:
                        print("End of training")
                        eot = True
                        torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_final.pt')
                        print("early stopping")

                else:
                    wait = 0

            if (epoch == self.args.epochs - 1):
                torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_final.pt')

            if (epoch % 50 == 0):
                torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_snap.pt')

        plt.plot(epoch_loss)
        print(epoch_loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title(f"Training loss for {self.args.model}")
        plt.savefig(f'./{self.args.model}/{self.args.model}_loss.png', bbox_inches="tight", pad_inches=0.0)
        plt.clf()

        # obs = torch.from_numpy(self.train.to_numpy()[0])
        # print(obs)
        # writer.add_graph(MemAE(),obs.float(),use_strict_trace=False)
        # writer.close()
        writer.flush()


    def define_model(self,trial):
        match self.args.model:
            case "AE":
                return build_base_AE(trial)

            case "DAE":
                return build_base_AE(trial)


            case "MAE":
                return build_MemAE(trial)


            case "VAE":

                hidden_dim = trial.suggest_int("hidden_dim", 3, 5)
                latent_dim = trial.suggest_int("latent_dim", 3, 5)
                return VAE(build_Encoder(5,hidden_dim,latent_dim),build_Decoder(latent_dim,hidden_dim,5))


            case "MVAE":
                hidden_dim = trial.suggest_int("hidden_dim", 3, 5)
                latent_dim = trial.suggest_int("latent_dim", 3, 5)
                mem_dim = trial.suggest_int("mem_dim", 40, 300)
                return MVAE(build_Encoder(5, hidden_dim, latent_dim), build_Decoder(latent_dim,hidden_dim,5),mem_dim,fe_dim=latent_dim, shrink_thres=0.0025)



    def objective(self,trial):

        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        self.optimizer = getattr(optim, optimizer_name)(self.model_.parameters(), lr=lr)
        model=self.define_model(trial)
        wait = 0
        epoch_loss = []
        eot = False  # end of training
        for epoch in range(15):



            if (eot == True):
                break
            losses = 0

            print(f"epoch: {epoch}")

            iteration = len(self.train) // self.args.batch
            for i in range(iteration):
                # latent_mae=[]
                # outputs = []

                obs = torch.from_numpy(self.train.iloc[i * self.args.batch:(i + 1) * self.args.batch].to_numpy())

                reconstructed = model(obs.float())
                #  print("obs")
                #  print(obs)
                # print("rec")
                # print(reconstructed['output'][0])

                # print(att_w)
                loss = self.loss_function(reconstructed['output'], obs.float())

                if self.mem:
                    att_w = reconstructed['att']
                    entropy_loss = self.tr_entropy_loss_func(att_w)
                    loss = loss + self.entropy_loss_weight * entropy_loss

                loss_val = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = losses + loss_val

            epoch_loss.append(losses / iteration)
            print(f"epoch_{epoch}_loss:  {losses / iteration}")

            if len(epoch_loss) > 2:
                if epoch_loss[epoch] > epoch_loss[epoch - 1]:
                    wait = wait + 1
                    if wait > self.args.patience:
                        print("End of training")
                        eot = True
                        torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_final.pt')
                        print("early stopping")

                else:
                    wait = 0

            if (epoch == self.args.epochs - 1):
                torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_final.pt')

            if (epoch % 50 == 0):
                torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_snap.pt')

        plt.plot(epoch_loss)
        print(epoch_loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title(f"Training loss for {self.args.model}")
        plt.savefig(f'./{self.args.model}/{self.args.model}_loss.png', bbox_inches="tight", pad_inches=0.0)
        plt.clf()
        return epoch_loss[-1]
    def reconstruct(self, dataframe, dataframe_ori, description="Reconstruction"):

        model_para = torch.load(f'./{self.args.model}/{self.args.model}_final.pt')
        self.model_.load_state_dict(model_para)
        print(self.model_)
        result = []  # reconstructed values
        errs = []
        for i in range(len(dataframe)):
            obs = torch.from_numpy(dataframe.iloc[i + 0:i + 1].to_numpy())
            reconstructed = self.model_(obs.float())
            result.append(reconstructed['output'].detach().numpy()[0])
            # latent.append(reconstructed['latent'].detach().numpy()[0])
            err= obs.float()-result[-1]
            errs.append(err.detach().numpy()[0])
            df_errs= pd.DataFrame(errs, columns=self.train.columns)
        # result= de_normalize_2d(np.array(result),self.norm)

        df_result = pd.DataFrame(result, columns=self.train.columns)
        df_result_with_err= df_result.copy()
        df_result_with_err['error']= errs
        df_result = de_normalize(df_result, dataframe_ori)
        dataframe = de_normalize(dataframe, dataframe_ori)

        if description == self.args.failure:
            self.pstart = self.args.fstart
            self.pstop = self.args.fstop

        for b in self.train.columns:
            plt.plot(df_result[b].iloc[self.pstart:self.pstop], linestyle='dotted', color='red',
                     label=f'Reconstructed_{self.args.model}',
                     marker='.')

            plt.plot(dataframe[b].iloc[self.pstart:self.pstop], label='Sensor data', color='blue', marker='.')

            if description == self.args.failure:
                plt.plot(dataframe_ori[b].iloc[self.pstart:self.pstop], label='Actual', color='black', marker='.')

            plt.legend()
            plt.xlabel(f"{b}")
            plt.title(description)
            plt.savefig(f'./{self.args.model}/{self.args.model}_{b}_{description}.jpg.png', bbox_inches="tight",
                        pad_inches=0.0)
            plt.clf()

        return df_result.iloc[self.pstart:self.pstop], dataframe.iloc[self.pstart:self.pstop] , df_errs #df_result_with_err

    def optimization(self): #https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
        print(f"Optimizing {self.args.model}")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=200)

        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")

        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


