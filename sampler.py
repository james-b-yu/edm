import numpy as np
import torch
import torch.nn.functional as F
# from utils import diffusion_utils
from noise_schedule import PredefinedNoiseSchedule
from model import EDM
from masked_model import MaskedEDM
# from masked_data import get_masked_qm9_dataloader
from model_config import get_config_from_args
import os



def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results

def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)






class Sampler():

    def __init__(self,args,dataloader,device,config=None):
        
        self.noise_schedule='cosine'
        self.noise_precision=1e-4
        self.timesteps = 1000
        self.T = 1000

        # Check these
        self.n_dims = 3
        self.num_classes = 5
        self.include_charges = True
        self.norm_values = (1., 1., 1.)
        self.norm_biases=(None, 0., 0.)
        self.parametrization = 'eps'


        # Load dataset (for shape and scaling reference)
        # dataloader = get_masked_qm9_dataloader(
        #     use_h=args.dataset == "qm9", split="test", batch_size=args.batch_size
        # )
        
        config = get_config_from_args(args, dataloader.dataset.num_atom_types)

        # config.schedule_type = 'cosine'

        self.dynamics = MaskedEDM(config)

        checkpoint_path = args.checkpoint
    
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            print("Loaded Checkpoint")
            # print(checkpoint.keys())
            self.dynamics.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("Checkpoint file not found!")
        
        self.dynamics.eval().to(device)

        print("Model Initialized")

        # I think this is wrong
        self.in_node_nf = 6

        self.gamma = PredefinedNoiseSchedule(self.noise_schedule, timesteps=self.timesteps,precision=self.noise_precision)

        print("Gamma Initialized")

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context,keep_frames=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """

        # print("Node Mask: ")
        # print(node_mask.size())

        # print("Edge Mask: ")
        # print(edge_mask.size())


        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        # print("Z Size: ")
        # print(z.size())

        # diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T

        # print("Self T: ")
        # print(self.T)
        
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            # print("Z Size 2: ")
            # print(z.size())

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context)
            


            # diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        # diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat
    
    def sample_combined_position_feature_noise(self,n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        # print("n_dims: ")
        # print(self.n_dims)
        # print("in_node_nf: ")
        # print(self.in_node_nf)

        z_x = self.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        
        z_h = self.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        
        z = torch.cat([z_x, z_h], dim=2)
    
        return z
    
    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        # print("Z Size 3: ")
        # print(zt.size())

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # # Ensure time_int is a torch.Tensor of type torch.long
        # time_int = torch.round(t * self.timesteps).long().to(zt.device)

        time_int=1/1000

        # print("Time Int: ")
        # print(time_int.size())

        # Use the get_eps_and_predicted_eps method to get the predicted epsilon values
        coord, one_hot, charge = zt[:, :, :3], zt[:, :, 3:-1], zt[:, :, -1:]

        print("Coord Size: ")
        print(coord.size())

        print("one_hot Size: ")
        print(one_hot.size())

        print("charge Size: ")
        print(charge.size())

        # # Ensure time_frac is a torch.Tensor of type torch.float
        # time_frac = t.float().to(zt.device)
        
        (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat) = self.dynamics.get_eps_and_predicted_eps(
            coord, one_hot, charge, time_int, node_mask, edge_mask
        )

        # print("Pred Eps Coord Size Orig: ")
        # print(pred_eps_coord.size())

        # print("Pred feat Coord Size Orig: ")
        # print(pred_eps_feat.size())
        # print(pred_eps_feat)

        pred_eps = torch.cat([pred_eps_coord, pred_eps_feat[:,:,:5], charge], dim=-1).to(pred_eps_coord.device)
        

        # print("Pred Eps Coord Size Extended: ")
        # print(pred_eps.size())

        # Ensure all tensors are on the same device
        zt = zt.to(pred_eps_coord.device)
        alpha_t_given_s = alpha_t_given_s.to(pred_eps_coord.device)
        sigma2_t_given_s = sigma2_t_given_s.to(pred_eps_coord.device)
        sigma_t = sigma_t.to(pred_eps_coord.device)
        sigma_s = sigma_s.to(pred_eps_coord.device)
        sigma_t_given_s = sigma_t_given_s.to(pred_eps_coord.device)

        # print("ZT Size: ")
        # print(zt.size())

        # print("Alpha T Given S Size: ")
        # print(alpha_t_given_s.size())

        # print("Sigma2 T Given S Size: ")
        # print(sigma2_t_given_s.size())

        # print("Sigma T Size: ")
        # print(sigma_t.size())

        # print("Pred Eps Coord Size: ")
        # print(pred_eps_coord.size())
        # print(sigma2_t_given_s.device)
        # print(sigma_t.device)
        # print(pred_eps_coord.device)

        # Compute mu for p(zs | zt)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * pred_eps

        # Compute sigma for p(zs | zt)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity
        zs = torch.cat([self.remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask), zs[:, :, self.n_dims:]], dim=2)
        
        return zs
    
    # def sample_p_zs_given_zt(self,s, t, zt, node_mask, edge_mask, context, fix_noise=False):
    #     """Samples from zs ~ p(zs | zt). Only used during sampling."""
    #     gamma_s = self.gamma(s)
    #     gamma_t = self.gamma(t)

    #     sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

    #     sigma_s = self.sigma(gamma_s, target_tensor=zt)
    #     sigma_t = self.sigma(gamma_t, target_tensor=zt)

    #     # Neural net prediction.
    #     eps_t = self.phi(zt, t, node_mask, edge_mask, context)

    #     # Compute mu for p(zs | zt).
    #     # diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
    #     # diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
    #     mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

    #     # Compute sigma for p(zs | zt).
    #     sigma = sigma_t_given_s * sigma_s / sigma_t

    #     # Sample zs given the paramters derived from zt.
    #     zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

    #     # Project down to avoid numerical runaway of the center of gravity.
    #     zs = torch.cat([self.remove_mean_with_mask(zs[:, :, :self.n_dims],node_mask),zs[:, :, self.n_dims:]], dim=2)
        
    #     return zs
    
    def remove_mean_with_mask(self,x, node_mask):
        masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
        assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
        N = node_mask.sum(1, keepdims=True)

        mean = torch.sum(x, dim=1, keepdim=True) / N
        x = x - mean * node_mask
        return x

    def sample_center_gravity_zero_gaussian_with_mask(self,size, device, node_mask):
        assert len(size) == 3
        x = torch.randn(size, device=device)

        x_masked = x * node_mask

        # This projection only works because Gaussian is rotation invariant around
        # zero and samples are independent!
        x_projected = self.remove_mean_with_mask(x_masked, node_mask)
        return x_projected

    def sample_gaussian_with_mask(self,size, device, node_mask):
        x = torch.randn(size, device=device)
        x_masked = x * node_mask
        return x_masked

    def sample_normal(self,mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        eps.to(mu.device)
        sigma= sigma.to(mu.device)
        
        return mu + sigma * eps
    
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s
    
    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        # net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        time_int=1/1000

        # print("Time Int: ")
        # print(time_int.size())

        # Use the get_eps_and_predicted_eps method to get the predicted epsilon values
        coord, one_hot, charge = z0[:, :, :3], z0[:, :, 3:-1], z0[:, :, -1:]

        # print("Coord Size: ")
        # print(coord.size())

        # print("one_hot Size: ")
        # print(one_hot.size())

        # print("charge Size: ")
        # print(charge.size())

        # # Ensure time_frac is a torch.Tensor of type torch.float
        # time_frac = t.float().to(zt.device)
        
        (eps_coord, eps_feat), (pred_eps_coord, pred_eps_feat) = self.dynamics.get_eps_and_predicted_eps(
            coord, one_hot, charge, time_int, node_mask, edge_mask
        )

        pred_eps = torch.cat([pred_eps_coord, pred_eps_feat[:,:,:5], charge], dim=-1).to(pred_eps_coord.device)


        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(pred_eps, z0, gamma_0)
        
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {'integer': h_int, 'categorical': h_cat}
        return x, h
    
    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)
    
    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)
    
    def phi(self, x, t, node_mask, edge_mask, context):
        # net_out = self.dynamics.forward(t, x, node_mask, edge_mask, context)
        net_out = self.dynamics.forward(node_mask=node_mask, edge_mask=edge_mask, time_frac=t, x=x)

        return net_out
    
    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)
    
    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output
    
    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int
    
    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out

            # Make sure on same device:

            zt = zt.to(eps_t.device)
            alpha_t = alpha_t.to(eps_t.device)
            sigma_t = sigma_t.to(eps_t.device)
            # eps_t = eps_t.to(eps_t.device)

            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred
    
############################
# Validity and bond analysis
def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm7b' or dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom':
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    return molecule_stable, nr_stable_bonds, len(x)

            

def sample(args,dataloader):

    print("Sampling Started")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using Device: ", device)

    sampler = Sampler(args,dataloader,device)
    print("Sampler Initialized")



    n_samples = 10
    n_nodes = 19
    n_tries = 1

    context = None

    node_mask = torch.ones(n_samples, n_nodes, 1).to(device)

    edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)

    # print(edge_mask)

    one_hot, charges, x = None, None, None
    for i in range(n_tries):
        chain = sampler.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100)
        chain = reverse_tensor(chain)

        # Repeat last frame to see final sample better.
        chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
        x = chain[-1:, :, 0:3]
        one_hot = chain[-1:, :, 3:-1]
        one_hot = torch.argmax(one_hot, dim=2)

        atom_type = one_hot.squeeze(0).cpu().detach().numpy()
        x_squeeze = x.squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataloader.dataset)[0]

        # Prepare entire chain.
        x = chain[:, :, 0:3]
        one_hot = chain[:, :, 3:-1]
        one_hot = F.one_hot(torch.argmax(one_hot, dim=2), num_classes=len(dataloader.dataset['atom_decoder']))
        charges = torch.round(chain[:, :, -1:]).long()

        if mol_stable:
            print('Found stable molecule to visualize :)')
            break
        elif i == n_tries - 1:
            print('Did not find stable molecule, showing last sample.')


    return one_hot, charges, x

    


if __name__ == "__main__":
    sample()

# TODO: fix device so ideally runs on CUDA rather than CPU