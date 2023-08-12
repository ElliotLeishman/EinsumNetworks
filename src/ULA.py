import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class welford: 
    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M)*(x - Mnext)
        self.M = Mnext
    
    def get_mean(self):
        return self.M
    
    def get_var(self):
        #when the number of samples is 1, we divide by self.k 
        if self.k < 2:
            return self.S/(self.k)
        else:
            return self.S/(self.k-1)

def max_eigenval(A, At, im_size, tol, max_iter, verbose, device):

    with torch.no_grad():

        #computes the maximum eigen value of the compund operator AtA
        
        x = torch.normal(mean=0, std=1,size=(im_size,im_size))[None][None].to(device)
        x = x/torch.norm(torch.ravel(x),2)
        init_val = 1
        
        for k in range(0,max_iter):
            y = A(x)
            x = At(y)
            val = torch.norm(torch.ravel(x),2)
            rel_var = torch.abs(val-init_val)/init_val
            if (verbose > 1):
                print('Iter = {}, norm = {}',k,val)
            
            if (rel_var < tol):
                break
            
            init_val = val
            x = x/val
        
        if (verbose > 0):
            print('Norm = {}', val)
        
        return val

def GradientIm(u, device):
    u_shapex = list(u.shape)
    u_shapex[0] = 1
    z = u[1:,:] - u[:-1,:]
    dux = torch.vstack([z, torch.zeros(u_shapex, device = device)])

    u_shapey = list(u.shape)
    u_shapey[1] = 1
    z = u[:,1:] - u[:,:-1]
    duy = torch.hstack([z, torch.zeros(u_shapey, device = device)])
    return dux, duy

def DivergenceIm(p1, p2):
    z = p2[:,1:-1] - p2[:,:-2]
    shape2 = list(p2.shape)
    shape2[1]=1
    v = torch.hstack([p2[:,0].reshape(shape2), z, -p2[:,-1].reshape(shape2)])
    
    shape1 = list(p1.shape)
    shape1[0]=1
    z = p1[1:-1,:] - p1[:-2,:]
    u = torch.vstack([p1[0,:].reshape(shape1), z, -p1[-1,:].reshape(shape1)])

    return v+u

def chambolle_prox_TV(g1, device, varargin):
  with torch.no_grad():

    g = g1.clone().detach()

    # initialize
    px = torch.zeros(g.shape, device = device)
    py = torch.zeros(g.shape, device = device)
    cont = 1     
    k    = 0

    #defaults for optional parameters
    tau = 0.249
    tol = 1e-3
    lambd = 1
    maxiter = 10
    verbose = 0
   
    #read the optional parameters
    for key in varargin.keys():
        if key.upper() == 'LAMBDA':
            lambd = varargin[key]
        elif key.upper() == 'VERBOSE':
            verbose = varargin[key]
        elif key.upper() == 'TOL':
            tol = varargin[key]
        elif key.upper() == 'MAXITER':
            maxiter = varargin[key]
        elif key.upper() == 'TAU':
            tau = varargin[key]
        elif key.upper() == 'DUALVARS':
            M,N = g.shape
            Maux, Naux = varargin[key].shape
            if M != Maux or N != 2*Naux:
                print('Wrong size of the dual variables')
                return
            px = torch.tensor(varargin[key])
            py = px[:,M:]
            px = px[:, 1:M]
        else:
            pass

    ## Main body
    while cont:
      k = k+1
      # compute Divergence of (px, py)
      divp = DivergenceIm(px,py) 
      u = divp - torch.divide(g, lambd).to(device)
      # compute gradient of u
      upx,upy = GradientIm(u, device)

      tmp = torch.sqrt(upx*upx + upy*upy).to(device)
      #error
      x1 = -upx.reshape(-1,1) + tmp.reshape(-1,1) * px.reshape(-1,1)
      y1 = -upy.reshape(-1,1) + tmp.reshape(-1,1) * py.reshape(-1,1)
      err = torch.sqrt(torch.sum(x1**2 + y1**2))

      # update px and py
      px = torch.divide(px + tau * upx,1 + tau * tmp).to(device)
      py = torch.divide(py + tau * upy,1 + tau * tmp).to(device)
      # check of the criterion
      cont = ((k<maxiter) and (err>tol))

    if verbose:
      print(f'\t\t|=====> k = {k}\n')
      print(f'\t\t|=====> err TV = {round(err,3)}\n')

    return g - lambd * DivergenceIm(px,py)

def blur_operators(kernel_len, size, type_blur, device, var = None):

    nx = size[0]
    ny = size[1]
    if type_blur=='uniform':
        h = torch.zeros(nx,ny)
        lx = kernel_len[0]
        ly = kernel_len[1]
        h[0:lx,0:ly] = 1/(lx*ly)
        c =  np.ceil((np.array([ly,lx])-1)/2).astype("int64")
    if type_blur=='gaussian':
        if var != None:
            [x,y] = torch.meshgrid(torch.arange(-ny/2,ny/2),torch.arange(-nx/2,nx/2))
            h = torch.exp(-(x**2+y**2)/(2*var))
            h = h/torch.sum(h)
            c = np.ceil(np.array([nx,ny])/2).astype("int64") 
        else:
            print("Choose a variance for the Gaussian filter.")

    H_FFT = torch.fft.fft2(torch.roll(h, shifts = (-c[0],-c[1]), dims=(0,1))).to(device)
    HC_FFT = torch.conj(H_FFT).to(device)

    # A forward operator
    A = lambda x: torch.fft.ifft2(torch.multiply(H_FFT,torch.fft.fft2(x))).real.reshape(x.shape)

    # A backward operator
    AT = lambda x: torch.fft.ifft2(torch.multiply(HC_FFT,torch.fft.fft2(x))).real.reshape(x.shape)

    AAT_norm = max_eigenval(A, AT, nx, 1e-4, int(1e4), 0, device)

    return A, AT, AAT_norm


def cshift(x,L):

    with torch.no_grad():

        N = len(x)
        y = torch.zeros(N)
        
        if L == 0:
            y = x.clone().detach()
            return y
        
        if L > 0:
            y[L:] = x[0:N-L]
            y[0:L] = x[N-L:N]
        else:
            L=int(-L)
            y[0:N-L] = x[L:N]
            y[N-L:N] = x[0:L]
            
        return y           
   



def tv(Dx):
    
    with torch.no_grad():

        Dx=Dx.view(-1)
        N = len(Dx)
        Dux = Dx[:int(N/2)]
        Dvx = Dx[int(N/2):N]
        tv = torch.sum(torch.sqrt(Dux**2 + Dvx**2))
        
        return tv

def Grad_Image(x, device):

    with torch.no_grad():

        x = x.to(device).clone()
        x_temp = x[1:, :] - x[0:-1,:]
        dux = torch.cat((x_temp.T,torch.zeros(x_temp.shape[1],1,device=device)),1).to(device)
        dux = dux.T
        x_temp = x[:,1:] - x[:,0:-1]
        duy = torch.cat((x_temp,torch.zeros((x_temp.shape[0],1),device=device)),1).to(device)
        return  torch.cat((dux,duy),dim=0).to(device)

