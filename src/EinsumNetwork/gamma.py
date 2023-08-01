import EinsumNetwork.ExponentialFamilyArray
import torch


class GammaArray(ExponentialFamilyArray):
    """Implementation of Gamma distribution."""

    def __init__(self, num_var, num_dims, array_shape, min_var=0.0001, max_var=10., use_em=True):
        super(GammaArray, self).__init__(num_var, num_dims, array_shape, 2 * num_dims, use_em=use_em)

    # Need to fix this! - Do I actually need to change anything because it is just intilising?
    def default_initializer(self):
        phi = torch.empty(self.num_var, *self.array_shape, 2*self.num_dims)
        with torch.no_grad():
            phi[..., 0:self.num_dims] = torch.randn(self.num_var, *self.array_shape, self.num_dims)
            phi[..., self.num_dims:] = 1. + phi[..., 0:self.num_dims]**2
        return phi

    # Projects parameters to constrained domain. In our case 
    def project_params(self, phi):
        phi_project = phi.clone()
        mu2 = phi_project[..., 0:self.num_dims] ** 2
        phi_project[..., self.num_dims:] -= mu2
        phi_project[..., self.num_dims:] = torch.clamp(phi_project[..., self.num_dims:], self.min_var, self.max_var)
        phi_project[..., self.num_dims:] += mu2
        return phi_project

    def reparam_function(self):
        def reparam(params_in):
            mu = params_in[..., 0:self.num_dims].clone()
            var = self.min_var + torch.sigmoid(params_in[..., self.num_dims:]) * (self.max_var - self.min_var)
            return torch.cat((mu, var + mu**2), -1)
        return reparam


    # Done
    def sufficient_statistics(self, x):
        if len(x.shape) == 2:
            stats = torch.stack((x, torch.log(x)), -1)
        elif len(x.shape) == 3:
            stats = torch.cat((x, torch.log(x)), -1)
        else:
            raise AssertionError("Input must be 2 or 3 dimensional tensor.")
        return stats


    def log_normalizer(self, theta):
        log_normalizer = -theta[..., 0:self.num_dims] ** 2 / (4 * theta[..., self.num_dims:]) - 0.5 * torch.log(-2. * theta[..., self.num_dims:])
        log_normalizer = torch.sum(log_normalizer, -1)
        return log_normalizer

    # Done
    def log_h(self, x):
        return 1