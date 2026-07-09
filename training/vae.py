"""
Modelo VAE
"""
import h5py
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.regression import R2Score 
from torch.utils.data import Dataset

beta = 0.0001
list_train_loss, list_val_loss = [],[]
list_val_rmse, list_val_r2, list_val_acc, list_val_ec, list_val_KL = [], [], [],[], []

#******************************************************************************************#
#******************************************************************************************#
#******************************************************************************************#
class Dataset(Dataset):
  #Loading data
  def __init__(self, x_true, y_label): #(batchsize, leads, samples) #(batchsize,)
    self.x_true = x_true
    self.label = y_label

  def __len__(self):
    return len(self.x_true)
  
  def __getitem__(self, i):
    self.x_true = self.x_true.reshape(self.x_true.shape[0], -1) #(batchsize, num_leads * frequency)
    target_in_true = torch.tensor(self.x_true[i], dtype=torch.float) #torch.Size([6, 8])
    target_out = torch.tensor(self.label[i], dtype=torch.int) #torch.Size([6, 8])
    return target_in_true,target_out
  
def rmse_loss(pred,true):
    "Computes the root mean square error"
    criterion = torch.nn.MSELoss()
    rmse = torch.sqrt(criterion(pred,true)) 
    return rmse

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
            
class encoder(torch.nn.Module):
    def __init__(self,num_features,latent_dim):
        super(encoder, self).__init__() #Its input is a datapoint xx
        label_dim = 1 #one label
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gru_01 = nn.Sequential(nn.GRU(input_size=num_features+label_dim, hidden_size=128,
                                           num_layers=4,batch_first=True, dropout = 0.3, device=device))
        self.densa_features= nn.Sequential(nn.Linear(in_features=128,out_features=64,bias=True, device=device),
                                           nn.LeakyReLU(0.3),
                                           nn.Linear(in_features=64,out_features=32,bias=True, device=device),
                                           nn.LeakyReLU(0.3),

                                           )
        self.gru_01.apply(init_weights)
        self.densa_features.apply(init_weights)
        # distribution parameters
        self.densa_mu = nn.Linear(in_features=32,out_features=latent_dim, device=device)
        self.densa_logvar = nn.Linear(in_features=32,out_features=latent_dim, device=device)

    def forward(self, in_data, in_labels): 
        """
        Its output is a hidden representation z
        From [batch_size, num_features] to [batch_size, latent_dim]
        Args:
            in_data = [batch_size, num_features] #features
            in_labels = [batch_size, 1] #labels
        Output:
            mu = [batch_size, latent_dim]
            variance = [batch_size, latent_dim]
        """
        device = in_data.device
        in_labels = torch.reshape(in_labels, (-1, 1)) #[32, 1]
        data = torch.cat([in_data, in_labels], dim=1).to(device) #[32, 49153]
        out_enc, _ = self.gru_01(data) #output:[batch_size, 128]
        bn = nn.BatchNorm1d(out_enc.shape[1]).to(device)
        out_enc = bn(out_enc) #[32, 128]
        out_enc = self.densa_features(out_enc) #[32, 32]
        mu = self.densa_mu(out_enc) #[32, 16]
        logvar = self.densa_logvar(out_enc) #[32, 16]
        return mu, logvar
    
class decoder(torch.nn.Module):
    def __init__(self, latent_dim,in_size):
        super(decoder, self).__init__()
        label_dim = 1 #número de clases
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gru_01 = nn.Sequential(nn.GRU(input_size=latent_dim+label_dim, hidden_size=32,
                                           num_layers=2,batch_first=True, device=device))
        self.densa_features= nn.Sequential(nn.Linear(in_features=32,out_features=64,bias=True, device=device),
                                           nn.Linear(in_features=64,out_features=128,bias=True, device=device))
        self.output = nn.Sequential(nn.Linear(in_features = 128, out_features = in_size, device=device),
                                    nn.Tanh()) #Sigmoid()
        self.gru_01.apply(init_weights)
        self.densa_features.apply(init_weights)
        self.output.apply(init_weights)

    def forward(self, z, in_labels): 
        """
        Its output is a x_hat and y_hat
        Args:
            z = [batch_size, latent_dim] #features
            in_labels = [batch_size, 1] #labels
        Output:
            x_hat = [batch_size, num_features]
            y_hat = [batch_size, 1]
        """
        in_labels = torch.reshape(in_labels, (-1, 1)) #[32, 1]
        z_concat = torch.cat([z, in_labels], dim=1) #50,66
        out_dec, _ = self.gru_01(z_concat) #[32, 32]
        out_dec = self.densa_features(out_dec) #[32, 128]
        out_dec = self.output(out_dec) #[32, 49152]
        return out_dec
    
class VAE(pl.LightningModule):
    def __init__(self, num_features,latent_dim,):
        super(VAE, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder(num_features,latent_dim)
        self.decoder = decoder(latent_dim,num_features)
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.validation_step_outputs_rmse = []
        self.validation_step_outputs_r2 = []
        self.validation_step_outputs_acc = []
        self.validation_step_outputs_recon_loss = []
        self.validation_step_outputs_kl_loss = []

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def reparameterize(self,mu,logvar,device):
        """
        sample z from q
        mu: (Tensor) Mean of the latent Gaussian [B x D]
        logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        Args:
            mu = [batch_size, latent_dim]
            logvar = [batch_size, latent_dim]
        Output:
            z = [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar).to(device)
        epsilon = torch.randn(mu.size()).to(device)
        z = mu + epsilon * std
        return z

    def forward(self,in_data, in_labels): #torch.Size([batchsize, num_leads * samples]) torch.Size([batchsize])
        device = in_data.device
        mu, logvar = self.encoder(in_data, in_labels)
        z = self.reparameterize(mu, logvar,device)
        x_recons = self.decoder(z,in_labels) # Reconstrucción
        return x_recons, mu, logvar, z
   
    def funcion_perdida(self,x_hat, x_true,mu, logvar,acc):
        device = x_hat.device
        recon_loss = nn.MSELoss()(x_hat, x_true).to(device)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()).to(device) ## KL Divergence Loss
        # loss = recon_loss + kl_loss * (1-acc)
        loss = recon_loss + beta * kl_loss
        return loss, recon_loss,kl_loss, acc          
            
    def training_step(self, batch):
        target_in_true,labels = batch
        x_synth, mu, logvar, z = self.forward(target_in_true, labels)
        loss, _,_,_ = self.funcion_perdida(x_synth,target_in_true,mu, logvar, 0)
        self.log("train_loss",loss)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self,batch,batch_idx):  
        """
        target_in_miss = [batch_size, leads * samples]
        target_in_true = [batch_size, leads * samples]
        mask = [batch_size, leads * samples]
        labels = [batch_size, num_classes]
        """
        target_in_true,labels = batch
        x_synth, mu, logvar, z = self.forward(target_in_true, labels)
        loss, recon_loss,kl_loss, acc = self.funcion_perdida(x_synth, target_in_true, mu, logvar, 0)
        self.log("val_loss",loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("recon_loss",recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("kl_loss",kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append(loss)
        self.validation_step_outputs_recon_loss.append(recon_loss)
        self.validation_step_outputs_kl_loss.append(kl_loss)

        rmse = rmse_loss(x_synth, target_in_true)
        self.validation_step_outputs_rmse.append(rmse)
        self.log("val_rmse",rmse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        r2score = R2Score().to(x_synth.device)
        self.validation_step_outputs_r2.append(r2score(x_synth.flatten(),target_in_true.flatten()))
        return {"val_r2":loss,"recon_loss":recon_loss,"kl_loss":kl_loss,"val_acc":acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer,"monitor": "val_loss"} 
    
    def on_train_epoch_end(self):
        #Obtaing ELBO loss
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_mean", epoch_mean)
        list_train_loss.append(np.mean(epoch_mean.item()))
        # # free up the memory
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        #Obtaing ELBO loss
        epoch_mean = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_mean", epoch_mean)
        list_val_loss.append(np.mean(epoch_mean.item()))
        self.validation_step_outputs.clear()

        #Obtaing recon_loss
        epoch_er_mean = torch.stack(self.validation_step_outputs_recon_loss).mean()
        self.log("validation_epoch_er_mean", epoch_mean)
        list_val_ec.append(np.mean(epoch_er_mean.item()))
        self.validation_step_outputs_recon_loss.clear()

        #Obtaing KL
        epoch_KL_mean = torch.stack(self.validation_step_outputs_kl_loss).mean()
        self.log("validation_epoch_KL_mean", epoch_KL_mean)
        list_val_KL.append(np.mean(epoch_KL_mean.item()))
        self.validation_step_outputs_kl_loss.clear()

        #Obtaing RMSE loss
        epoch_rmse_mean = torch.stack(self.validation_step_outputs_rmse).mean()
        self.log("validation_epoch_rmse_mean", epoch_rmse_mean)
        list_val_rmse.append(np.mean(epoch_rmse_mean.item()))
        self.validation_step_outputs_rmse.clear()

        #Obtaing R2 loss
        epoch_r2_mean = torch.stack(self.validation_step_outputs_r2).mean()
        self.log("validation_epoch_r2_mean", epoch_r2_mean)
        list_val_r2.append(np.mean(epoch_r2_mean.item()))
        self.validation_step_outputs_r2.clear()

 
def saving_hdf5(datos, path, filename):
    with h5py.File(path+filename+".hdf5", 'w') as f:
        f.create_dataset('tracings', data=datos)
    print("Saved: (hdf5):", filename)

def saving_labels(etiquetas, path,filename):
    pd_label = pd.DataFrame(etiquetas, columns= ["Label"])
    if 'Unnamed: 0' in pd_label.columns:
        pd_label = pd_label.drop(['Unnamed: 0'], axis = 1)
    pd_label.to_csv(path+filename+"_labels.csv")


def dwprobability(test_true,test_pred,output_seq_length,num_features,fold,fs):
    """Función para calcular y graficar la dimension_wise_probability"""

    # test_true = test_true.cpu().detach().numpy()
    # test_pred = test_pred.cpu().detach().numpy()

    prob_real = np.mean(test_true, axis=0).reshape(-1)
    prob_syn = np.mean(test_pred, axis=0).reshape(-1)

    p1 = plt.scatter(prob_real, prob_syn, c ='b', alpha=0.5)    
    x_max = max(np.max(prob_real), np.max(prob_syn)) #0.9678864
    x_min = min(np.min(prob_real), np.min(prob_syn)) #x_min: 0.23113397
    x = np.linspace(x_min-0.5, x_max + 0.5)
    p2 = plt.plot(x, x, linestyle='--', color='gray', label="Ideal")  # solid
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.tick_params(labelsize=10)
    plt.legend(loc=2, prop={'size': 10})
    plt.title('Rendimiento de probabilidad por dimensión \n (datos reales vs datos predichos)\n')
    plt.xlabel('Datos reales')
    plt.ylabel('Predicción')
    plt.savefig("../images/augmentation/vae_dw_"+str(fold)+"_fs_"+fs+".svg")
    plt.close()

def plot_boxplot(test_true,test_pred,num_leads,feature_selection,fold):
    from sklearn.preprocessing import MinMaxScaler

    # Generate random sample data
    test_true = test_true.reshape(-1,num_leads)
    test_pred = test_pred.reshape(-1,num_leads)

    #Escalando datos:
    scaler = MinMaxScaler(feature_range=(0,1)).fit(test_true)
    test_true = scaler.transform(test_true)
    test_pred = scaler.transform(test_pred)

    #print("test_true: ",test_true.shape, "test_pred: ",test_pred.shape, feature_selection, type(feature_selection)) #test_true:  (12000, 5) test_pred:  (12000, 5) si <class 'str'>

    if feature_selection == "si":
        lead_names = ['AVR', 'DI', 'DII', 'V1', 'AVF']
        lead_names_aug = ['AVR_aug', 'DI_aug', 'DII_aug', 'V1_aug', 'AVF_aug']
    else:
        lead_names = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"] # mismo que samitrop!
        lead_names_aug = ["DI_aug", "DII_aug", "DIII_aug", "AVR_aug", "AVL_aug", "AVF_aug", "V1_aug", "V2_aug", "V3_aug", "V4_aug", "V5_aug", "V6_aug"] # mismo que samitrop!
    

    test_true_pd = pd.DataFrame(test_true, columns = lead_names)
    test_pred_pd = pd.DataFrame(test_pred, columns = lead_names_aug)

    combined_df = pd.concat([test_true_pd, test_pred_pd], ignore_index=True)

    ########################
    #Ordenando las columnas
    if feature_selection == "si":
        combined_df = combined_df[['AVR','AVR_aug','DI','DI_aug','DII','DII_aug','V1','V1_aug','AVF','AVF_aug']]
    else:
        combined_df = combined_df[["DI","DI_aug", "DII","DII_aug","DIII","DIII_aug","AVR","AVR_aug","AVL","AVL_aug",
                                   "AVF","AVF_aug","V1","V1_aug","V2","V2_aug","V3","V3_aug","V4", "V4_aug", "V5","V5_aug","V6","V6_aug"]]
    #########################

    boxplot = combined_df.boxplot(grid=True, rot=45, fontsize=8, figsize=(10, 7), showfliers=False)
    plt.xlabel("Leads",fontsize=8)
    plt.ylabel("Values",fontsize=8)
    # plt.ylim(0,1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("../images/augmentation/vae_boxplot_"+str(fold)+"_fs_"+feature_selection+".svg")
    plt.close()

def plot_4paneles(test_true,test_pred,num_leads,feature_selection,fold):
    from sklearn.preprocessing import MinMaxScaler

    if feature_selection == "si":
        lead_names = ['AVR', 'DI', 'DII', 'V1', 'AVF']
        lead_names_aug = ['AVR_aug', 'DI_aug', 'DII_aug', 'V1_aug', 'AVF_aug']
    else:
        lead_names = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"] # mismo que samitrop!
        lead_names_aug = ["DI_aug", "DII_aug", "DIII_aug", "AVR_aug", "AVL_aug", "AVF_aug", "V1_aug", "V2_aug", "V3_aug", "V4_aug", "V5_aug", "V6_aug"] # mismo que samitrop!
    

    SAMPLING_RATE = 400
    print("test_true: ", test_true.shape) #3, 4000, 5
    print("test_pred: ", test_pred.shape) #3, 4000, 5

    test_true_ind = test_true[0, :, :] #(4000, 5)
    test_pred_ind = test_pred[0, :, :] #(4000, 5)
    print("test_true_ind: ", test_true_ind.shape, "test_pred_ind: ",test_pred_ind.shape) 

    test_true_ind = np.transpose(test_true_ind, axes=(1,0)) #(5, 4000)
    test_pred_ind = np.transpose(test_pred_ind, axes=(1,0)) #(5, 4000)
    print("test_true_ind: ", test_true_ind.shape, "test_pred_ind: ",test_pred_ind.shape) #(5, 4000)


    print("\n *********** Real vs predicted *******************")
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    t = np.arange(test_true_ind.shape[1]) / SAMPLING_RATE
    for i, ax in enumerate(axes):
        ax.plot(t, test_true_ind[i], lw=0.7, color='blue', label = "Real")
        ax.plot(t, test_pred_ind[i], lw=0.7, color='red', label = "Synthetic")
        ax.set_ylabel(lead_names[i], fontsize=10, rotation=0, labelpad=20)
        ax.set_yticks([])
        ax.spines[['top','right','left']].set_visible(False)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=2)
    plt.savefig("../images/augmentation/vae_1signal_"+str(fold)+"_fs_"+feature_selection+".svg")


    print("\n *********** KDE *******************")
    fig, ax = plt.subplots(figsize=(8, 5)) 
    sns.kdeplot(test_true.flatten(), ax = ax, fill=True, color='blue', label = "Real")
    sns.kdeplot(test_pred.flatten(), ax = ax, fill=True, color='red', label = "Synthetic")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.),fancybox=True, shadow=True, ncol=2)
    plt.savefig("../images/augmentation/vae_2KDE_"+str(fold)+"_fs_"+feature_selection+".svg")

    print("\n *********** PCA *******************")

    # -----------------------------
    # 1. Preparar datos
    # -----------------------------
    X_real_flat = test_true.reshape(test_true.shape[0], -1)
    X_synth_flat = test_true.reshape(test_true.shape[0], -1)

    X = np.vstack([X_real_flat, X_synth_flat])
    print("X: ", X.shape)

    labels = np.array([0] * len(test_true) +   # real
                      [1] * len(test_true)    # sintético
                    )
    
    # -----------------------------
    # 2. Escalado (IMPORTANTE para t-SNE)
    # -----------------------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # -----------------------------
    # 3. t-SNE
    # -----------------------------

    from sklearn.decomposition import PCA
    X_pca50 = PCA(n_components=5).fit_transform(X_scaled)
    #X_tsne = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(X_pca50)

    tsne = TSNE(
        n_components=2,
        perplexity=5,   # prueba 5–50 según dataset
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_pca50)

    # -----------------------------
    # 4. Visualización
    # -----------------------------
    plt.figure(figsize=(8, 6))

    plt.scatter(X_tsne[labels == 0, 0], X_tsne[labels == 0, 1], color = "blue",  label="Real", alpha=0.6, s=20)
    plt.scatter(X_tsne[labels == 1, 0], X_tsne[labels == 1, 1], color = "red", label="Synthetic", alpha=0.6, s=20)
    plt.title("t-SNE: ECG real vs sintético")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.),fancybox=True, shadow=True, ncol=2)
    plt.savefig("../images/augmentation/vae_3TSNE_"+str(fold)+"_fs_"+feature_selection+".svg")

def data_augmentation_method(model, data_loader, device, num_leads, augmentation_rate,feature_selection):
    """
    Función para hacer aumentado de datos
    """
    model.eval()
    model.to(device)
    xhat_list, label_list, xoriginal_list, outputs = [],[],[],[]
    with torch.no_grad(): #turn off gradients computation
        for i, datos in enumerate(data_loader):
            datos, etiqueta  = datos
            datos = datos.detach().clone().to(device)
            etiqueta = etiqueta.detach().clone().to(device)
            x_augmentados,_,_,_= model(datos, etiqueta)

            # print("datos originales: ", datos.shape) #3, 20000
            # print("datos aumentados: ", x_augmentados.shape) #3, 20000
            # dwprobability(datos, x_augmentados, 4000, num_leads, augmentation_rate, "Per batch") #visualiza por batch

            #- Cálculo para gráficas de TSNE
            mu, logvar = model.encoder(datos, etiqueta)
            z = model.reparameterize(mu, logvar, device)
            outputs.append(z)
            #----
            
            xhat_list.extend(x_augmentados.cpu().detach().numpy().reshape(x_augmentados.shape[0], -1))
            xoriginal_list.extend(datos.cpu().detach().numpy().reshape(datos.shape[0], -1))
            label_list.extend(etiqueta.cpu().detach().numpy())
        
        np_xhat = np.array(xhat_list)#.reshape(-1, 4000, num_leads)
        np_xoriginal = np.array(xoriginal_list)

        ##### Visualizando datos aumentados
        dwprobability(np_xoriginal, np_xhat, 4000, num_leads, augmentation_rate, feature_selection)

        np_xhat = np.array(xhat_list).reshape(-1, 4000, num_leads)
        np_xoriginal = np_xoriginal.reshape(-1, 4000, num_leads)
        plot_boxplot(np_xoriginal,np_xhat,num_leads,feature_selection,augmentation_rate)
        plot_4paneles(np_xoriginal,np_xhat,num_leads,feature_selection,augmentation_rate)

        return np_xhat
