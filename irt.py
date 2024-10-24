import pickle
import numpy as np
import math
from tqdm.auto import tqdm
import torch
import os
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering

sigmoid = nn.Sigmoid()
np_sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-30,30)))
np_logit = lambda x: np.log(x/(1-x))

def create_irt_train_mask(Y, validation_fraction, random_seed):
    mask = torch.ones(Y.shape, dtype=torch.bool).reshape(-1)
    local_state = np.random.RandomState(random_seed)
    mask[local_state.choice(len(mask), int(validation_fraction*len(mask)+1))] = False
    mask = mask.reshape(Y.shape)
    return mask

def loss_matrix(Y, P, eps=1e-5):
    return -(Y*(P+eps).log() + (1-Y)*(1-P+eps).log())

def np_irt_forward(Theta, Alpha, beta, kappa, max_asymp = 1):
    return max_asymp*kappa.T + (1-max_asymp*kappa.T)*np_sigmoid(Theta@Alpha.T-beta.T)
    
def irt_forward(Theta, Alpha, beta, kappa, max_asymp = 1):
    return max_asymp*kappa.T + (1-max_asymp*kappa.T)*sigmoid(Theta@Alpha.T-beta.T)
    
def fit_IRT(Y,
            d,
            log_Alpha=None,
            beta=None,
            Theta=None,
            logit_kappa=None,
            lr = 1,
            n_epochs = 10000,
            validation_fraction=.1,
            tol = 1e-5,
            reg = 0,
            model='m3pl',
            early_stop_patience = 50,
            scheduler_factor = 0.9,
            scheduler_patience = 10,
            val_step = 2,
            print_every = 100,
            random_seed = 42,
            verbose = False,
            device='cpu'):

    assert val_step<=print_every
    assert model in ['m2pl','m3pl']

    if model=='m3pl': max_asymp = 1
    else: max_asymp = 0
        
    ### Basic defs
    if validation_fraction>0:
        train_mask = create_irt_train_mask(Y, validation_fraction, random_seed)
        val_mask = ~train_mask
    Y = torch.tensor(Y, requires_grad=False).to(device)
    n_llms = Y.shape[0]

    ### Defining training variables
    parameters = []
    torch.manual_seed(random_seed)
    
    #beta
    if beta is None:
        beta = torch.nn.Parameter(torch.normal(0, .1, size=(Y.shape[1],1,), dtype=torch.float32, device=device))
        parameters.append(beta)
    else:
        beta = torch.tensor(beta, requires_grad=False).float().to(device)

    #kappa
    if logit_kappa is None:
        logit_kappa = torch.nn.Parameter(torch.normal(0, .1, size=(Y.shape[1],1,), dtype=torch.float32, device=device))
        parameters.append(logit_kappa)
    else:
        logit_kappa = torch.tensor(logit_kappa, requires_grad=False).float().to(device)
        
    #Alpha
    if log_Alpha is None:
        scale = np.sqrt(np.log((1+(1+4/d**.5)**.5)/2)) #guarantees std(Alpha)~1/sqrt(d)
        log_Alpha = torch.nn.Parameter(torch.normal(0, scale, size=(Y.shape[1],d,), dtype=torch.float32, device=device)) 
        parameters.append(log_Alpha)
    else:
        log_Alpha = torch.tensor(log_Alpha, requires_grad=False).float().to(device)

    #Theta
    if Theta is None:
        Theta = torch.nn.Parameter(torch.normal(0, 1/(d**.5), size=(n_llms,d,), dtype=torch.float32, device=device))
        parameters.append(Theta)
    else:
        Theta = torch.tensor(Theta, requires_grad=False).float().to(device)
    
    ### Training
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    
    # Early stopping parameters
    best_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    early_stop = False
    
    # Losses
    train_losses =[]
    val_losses =[]
    val_accs =[]

    ### Defining sub-function for optimization
    def get_reg(beta, Alpha, Theta):
        return reg*(((beta**2).mean() + (Alpha**2).mean() + (Theta**2).mean()))

    for epoch in tqdm(range(n_epochs), disable=not verbose):
        optimizer.zero_grad()
        P = irt_forward(Theta, log_Alpha.exp(), beta, sigmoid(logit_kappa), max_asymp=max_asymp)
        regularizer = get_reg(beta, log_Alpha.exp(), Theta)
        regularizer_value = regularizer.item()
        if validation_fraction>0: loss = loss_matrix(Y, P)[train_mask].mean() + regularizer
        else: loss = loss_matrix(Y, P).mean() + regularizer
        loss.backward()
        optimizer.step()
    
        train_losses.append(loss.item() - regularizer_value)

        if epoch%val_step==0:
            with torch.no_grad():
                if validation_fraction>0:
                    P = irt_forward(Theta, log_Alpha.exp(), beta, sigmoid(logit_kappa), max_asymp=max_asymp)
                    loss = loss_matrix(Y, P)[val_mask].mean()
                    val_losses.append(loss.item())
                    val_accs.append((Y==(P>.5))[val_mask].float().mean().item())
                    scheduler.step(val_losses[-1])
        
                    # Check for early stopping
                    if val_losses[-1] + tol < best_loss:
                        best_loss = val_losses[-1]
                        best_epoch = epoch
                        epochs_no_improve = 0

                        best_beta = beta.detach().cpu().numpy()
                        best_log_Alpha = log_Alpha.detach().cpu().numpy()
                        best_Theta = Theta.detach().cpu().numpy()
                        best_logit_kappa = logit_kappa.detach().cpu().numpy()
                    else:
                        epochs_no_improve += 1
        
                    if epochs_no_improve >= early_stop_patience:
                        if verbose: print(f"Early stop at epoch {epoch} - best val loss {best_loss:.5f} - best epoch {best_epoch}")
                        early_stop = True
                        break

                    if verbose:
                        if epoch%print_every==0:
                            tqdm.write(f"epoch={epoch:04d}, d={d}, reg={reg}, tol={tol}, train loss={train_losses[-1]:.5f}, val loss={val_losses[-1]:.5f}, val acc={val_accs[-1]:.5f}, lr={scheduler.optimizer.param_groups[0]['lr']:.5f}")
                
                else:
                     # Check for early stopping
                    scheduler.step(train_losses[-1])
                    if train_losses[-1] + tol < best_loss:
                        best_loss = train_losses[-1]
                        best_epoch = epoch
                        epochs_no_improve = 0

                        best_beta = beta.detach().cpu().numpy()
                        best_log_Alpha = log_Alpha.detach().cpu().numpy()
                        best_Theta = Theta.detach().cpu().numpy()
                        best_logit_kappa = logit_kappa.detach().cpu().numpy()
                    else:
                        epochs_no_improve += 1
        
                    if epochs_no_improve >= early_stop_patience:
                        if verbose: print(f"Early stop at epoch {epoch} - best train loss {best_loss:.5f} - best epoch {best_epoch}")
                        early_stop = True
                        break

                    if verbose:
                        if epoch%print_every==0:
                            tqdm.write(f"epoch={epoch:04d}, d={d}, reg={reg}, tol={tol}, train loss={train_losses[-1]:.5f}, lr={scheduler.optimizer.param_groups[0]['lr']:.5f}")

           
    ### Output
    beta = best_beta
    log_Alpha = best_log_Alpha
    Theta = best_Theta
    logit_kappa = best_logit_kappa
    
    if validation_fraction>0:
        val_loss = best_loss
    else:
        val_loss = None

    if model=='m3pl': return log_Alpha, beta, Theta, logit_kappa, val_loss
    else: return log_Alpha, beta, Theta, None, val_loss


class IRT:
    def __init__(self, ds=[1,2,3,5,10,15], device='cpu'):

        self.ds = ds
        self.device = device
        self.model = 'm3pl' #the IRT class might not work if self.model != 'm2pl' because self.kappa = None
 
    def fit(self,
            Y,
            lr=1,
            n_epochs=10000,
            validation_fraction=.05,
            tol=1e-5,
            regs=[1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 1e-10],
            random_seed=42,
            standard=True,
            verbose=True):

        assert validation_fraction>0 and validation_fraction<1

        self.best_loss = math.inf
    
        for d in tqdm(self.ds): #, disable=not verbose
            best_loss_reg = math.inf
            bad_d = True
            
            for reg in regs:
                
                log_Alpha, beta, Theta, logit_kappa, val_loss = fit_IRT(Y,
                                                                        d,
                                                                        log_Alpha=None,
                                                                        beta=None,
                                                                        Theta=None,
                                                                        logit_kappa=None,
                                                                        model=self.model,
                                                                        lr=lr,
                                                                        n_epochs=n_epochs,
                                                                        validation_fraction=validation_fraction,
                                                                        tol=tol,
                                                                        reg=reg,
                                                                        random_seed=random_seed,
                                                                        verbose=verbose,
                                                                        device=self.device)


                if val_loss + tol < self.best_loss:
                    bad_d = False
                    self.d = d
                    self.reg = reg
                    self.beta = beta
                    self.Alpha = np.exp(log_Alpha)
                    self.Theta = Theta
                    if self.model=='m3pl':
                        self.kappa = np_sigmoid(logit_kappa)
                    else:
                        self.kappa = None
                    self.best_loss = val_loss  

                if val_loss + tol >= best_loss_reg: #for sake of efficiency (if we realize that further reducing regs won't help, we stop)
                    break
                else:
                    best_loss_reg = val_loss

            if bad_d:
                if verbose:
                    print(f"\nValidation loss of d={d} is not better than smaller d: we stop validation")
                break
        #if verbose: 
        print(f"\nBest d={self.d} - best val loss={self.best_loss:.5f} - best reg={self.reg}")

        if standard:
            self.stand(verbose=verbose)

    def stand(self, n_max = 5000, verbose=False):

        #standard
        if verbose:
            print(f"\nStandardizing Theta...")
        mu = self.Theta.mean(0)[None,:]
        sigma = self.Theta.std(0)[None,:]
        self.beta -= self.Alpha@mu.T
        self.Alpha *= sigma
        self.Theta = (self.Theta-mu)/sigma
        
        if self.d>1:
            #change of basis
            if verbose:
                print(f"\nClustering items...")
            n,d = self.Alpha.shape
            select = list(range(0,n,int(np.ceil(n/n_max))))
            A = self.Alpha[select]
            simil = cosine_similarity(A) 
            clustering = SpectralClustering(n_clusters=d, assign_labels='kmeans', affinity='precomputed', random_state=0).fit(simil)
            labels, counts = np.unique(clustering.labels_, return_counts=True)
            labels = labels[np.argsort(-counts)]
            counts = counts[np.argsort(-counts)]
            labels = np.array([np.argmax(labels==l) for l in clustering.labels_])
            if verbose:
                print(f"- relative frequency per cluster:{counts/counts.sum()}")
                print(f"- check self.Theta2 for abilities for specific clusters and self.U for directions.")
                
            U = []
            for l in range(d):
                A = self.Alpha[select][labels==l]
                v,u=np.linalg.eigh(A.T@A)
                U.append(np.abs(u[:,-1]))
            U = np.vstack(U).T
            all_labels = cosine_similarity(self.Alpha,U).argmax(1)
            all_labels[select] = labels
            self.clusters = all_labels
            _, counts = np.unique(all_labels, return_counts=True)
            R = np.linalg.inv(U.T)
            self.U,self.R=U,R
            #self.Alpha2 = self.Alpha@R
            self.Theta2 = self.Theta@np.linalg.inv(R).T #(=self.Theta@U)
            mu = self.Theta2.mean(0)[None,:]
            sigma = self.Theta2.std(0)[None,:]
            self.Theta2 = (self.Theta2-mu)/sigma

    def get_params(self):
        return self.Alpha, self.beta, self.kappa, self.Theta

    def fit_theta(self, Y, selected_items, lr=1, n_epochs=10000, tol=1e-5, random_seed = 42, verbose=True):
        
        _, _, Theta_test, _, _ = fit_IRT(Y,
                                         self.d,
                                         log_Alpha=np.log(self.Alpha[selected_items]),
                                         beta=self.beta[selected_items],
                                         Theta=None,
                                         logit_kappa=np_logit(self.kappa[selected_items]),
                                         model=self.model,
                                         lr=lr,
                                         n_epochs=n_epochs,
                                         validation_fraction=0,
                                         tol=tol,
                                         reg=self.reg,
                                         random_seed=random_seed,
                                         device=self.device,
                                         verbose=verbose)
        
        return {'new_Theta': Theta_test}

    def fit_alpha_beta_kappa(self, Y, selected_test_takers, lr=1, n_epochs=10000, tol=1e-5, random_seed = 42, verbose=True):
        
        log_Alpha, beta, _, logit_kappa, _ = fit_IRT(Y,
                                                     self.d,
                                                     log_Alpha=None,
                                                     beta=None,
                                                     Theta=self.Theta[selected_test_takers],
                                                     logit_kappa=None,
                                                     model=self.model,
                                                     lr=lr,
                                                     n_epochs=n_epochs,
                                                     validation_fraction=0,
                                                     tol=tol,
                                                     reg=self.reg,
                                                     random_seed=random_seed,
                                                     device=self.device,
                                                     verbose=verbose)

        return {'new_Alpha': np.exp(log_Alpha), 'new_beta': beta, 'new_kappa': np_sigmoid(logit_kappa)}

    def save(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump({
                'ds': self.ds,
                'device': self.device,
                'd': self.d,
                'reg': self.reg,
                'beta': self.beta,
                'Alpha': self.Alpha,
                'kappa': self.kappa,
                'Theta': self.Theta,
                'best_loss': self.best_loss
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.ds = params['ds']
            self.device = params['device']
            self.d = params['d']
            self.reg = params['reg']
            self.beta = params['beta']
            self.Alpha = params['Alpha']
            self.kappa = params['kappa']
            self.Theta = params['Theta']
            self.best_loss = params['best_loss']
