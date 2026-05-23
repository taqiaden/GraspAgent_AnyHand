import numpy as np
import torch
import torch.nn.functional as F
from Configurations.config import device

def generate_random_beta_dist_widh( size):
    sampled_approach = (torch.rand((size, 2), device=device) - 0.5)  # *1.5
    ones_ = torch.ones_like(sampled_approach[:, 0:1])
    sampled_approach = torch.cat([sampled_approach, ones_], dim=1)

    verticle = torch.zeros((size, 3), device=device)
    verticle[:, -1] += 1
    # sampled_approach=verticle
    sampled_approach = sampled_approach * 0.5 + verticle * 0.5
    sampled_approach = F.normalize(sampled_approach, dim=-1)

    sampled_beta = (torch.rand((size, 2), device=device) - 0.5) * 2
    sampled_beta = F.normalize(sampled_beta, dim=1)


    sampled_dist = torch.rand((size, 1), device=device)**2


    sampled_width=1-torch.rand((size, 1), device=device)**2

    sampled_pose = torch.cat([sampled_approach, sampled_beta, sampled_dist, sampled_width], dim=1)

    return sampled_pose

def beta_peak_intensity_tensor(n, c, centers, data_range, peak_intensity=30.0):
    """
    Generate tensor with controllable peak intensity for each channel

    Args:
        n: number of samples per channel
        c: number of channels
        centers: tensor of size [c] with center for each channel
        data_range: (min, max) values
        peak_intensity:
            - 1.0: uniform distribution
            - 2-5: moderate peak
            - 5-10: strong peak
            - 10+: very sharp peak, minimal tails
    """
    min_val, max_val = data_range

    # print('centers: ',centers)
    # print('data_range: ',data_range)

    # Ensure centers is a tensor of correct shape
    if isinstance(centers, (list, np.ndarray)):
        centers = torch.tensor(centers, dtype=torch.float32,device=centers.device)
    assert centers.shape == (c,), f"Centers must have shape [c], got {centers.shape}"

    # Normalize centers to [0,1] within the range for each channel
    centers_norm = (centers - min_val) / (max_val - min_val)

    # Calculate Beta parameters for each channel
    # Shape: [c] for alpha and beta
    alpha = peak_intensity * centers_norm + 1
    beta = peak_intensity * (1 - centers_norm) + 1

    # Generate Beta distributed samples for each channel
    # We'll generate samples separately for each channel and then combine
    samples_list = []
    for i in range(c):
        # print((alpha[i], beta[i]))
        beta_dist = torch.distributions.Beta(alpha[i], beta[i])
        channel_samples = beta_dist.sample((n,))
        samples_list.append(channel_samples)

    # Stack along channel dimension to get [n, c]
    samples = torch.stack(samples_list, dim=1)

    # Scale to desired range
    tensor_data = samples * (max_val - min_val) + min_val

    return tensor_data

def quat_between_batch(v_from, v_to):
    """
    Compute quaternions to rotate a single vector v_from to each vector in v_to.

    Args:
        v_from: Tensor of shape [3], source vector.
        v_to: Tensor of shape [n, 3], target vectors.

    Returns:
        quats: Tensor of shape [n, 4], quaternions in [w, x, y, z] format.
    """
    # Normalize input vectors
    v_from = v_from / torch.norm(v_from)
    v_to = v_to / torch.norm(v_to, dim=1, keepdim=True)

    # Compute cross product and dot product
    cross = torch.cross(v_from.expand_as(v_to), v_to, dim=1)
    dot = torch.sum(v_to * v_from, dim=1, keepdim=True)

    # Compute quaternion scalar part
    w = torch.sqrt(torch.sum(v_from ** 2) * torch.sum(v_to ** 2, dim=1, keepdim=True)) + dot

    # Combine w and cross
    quat = torch.cat([w, cross], dim=1)

    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=1, keepdim=True)
    return quat

def random_unit_circle(n):
    # sample angles uniformly in [0, 2π)
    theta = 2 * torch.pi * torch.rand(n,device=device)

    # convert to unit vectors
    x = torch.cos(theta)
    y = torch.sin(theta)

    return torch.stack([x, y], dim=-1)

def generate_random_CH_poses(size):

    alpha_ = torch.cat([random_unit_circle(size),torch.rand((size,1),device=device)*-1],dim=1)
    alpha_ = F.normalize(alpha_, dim=-1)

    beta_ = random_unit_circle(size)
    # beta_[:,1]=beta_[:,1].abs()*-1

    # values = torch.tensor([-1., 0., 1.])
    # beta_=sample_vectors(size,2,values).to(device)
    beta_ = F.normalize(beta_, dim=-1)

    # values = torch.tensor([-0.5, 0.,0.2, 0.5])
    # fingers_=sample_vectors(size,3,values).to(device)

    delta = torch.randn((size, 3), device=device)/2
    delta[:,0:2]/=5
    delta[:,-1]-=0.5

    # values = torch.tensor([ 0.,0.3,0.5,0.7, 1.])
    # transition_=sample_vectors(size, 1, values).to(device)
    fingers = torch.randn((size, 3), device=device)+0.5

    sampled_pose = torch.cat([alpha_,beta_, delta, fingers], dim=1)
    return sampled_pose

def generate_random_SH_poses(size):

    alpha_ = torch.cat([random_unit_circle(size),torch.rand((size,1),device=device)*-1],dim=1)
    alpha_ = F.normalize(alpha_, dim=-1)

    beta_ = random_unit_circle(size)
    beta_ = F.normalize(beta_, dim=-1)

    fingers_ = beta_peak_intensity_tensor(size, 3, torch.tensor([0.,0,0]).to(device),[-0.5,0.5], peak_intensity=10.0)

    delta = torch.randn((size, 3), device=device)
    delta[:,0:2]/=5
    
    sampled_pose = torch.cat([alpha_,beta_,delta, fingers_], dim=1)
    return sampled_pose

def generate_random_SH_5F_poses(size):
    alpha_ = torch.cat([random_unit_circle(size),torch.rand((size,1),device=device)*-1],dim=1)
    alpha_ = F.normalize(alpha_, dim=-1)

    beta_ = random_unit_circle(size)
    beta_ = F.normalize(beta_, dim=-1)

    gamma = torch.randn((size,2),device=device)/3

    s = 1-torch.rand((size, 4), device=device)**2
    b=beta_peak_intensity_tensor(size, 5, torch.tensor([0.6,1.3,1.3,1.3,1.3]).to(device),[-0.262,1.57], peak_intensity=10.0)

    fingers_ = torch.randn((size, 18), device=device)/3

    fingers_[:,0:1]=(torch.rand((size, 1), device=device)-0.5)*2*1.05

    fingers_[:, 1:2]=(1-torch.rand((size, 1), device=device)**2)*1.2

    fingers_[:, 3:4] =beta_peak_intensity_tensor(size, 1, torch.tensor([0.698]).to(device),[-.7,0.7], peak_intensity=10.0)
    # fingers_[:, 3:4] =(torch.rand((size, 1), device=device)-0.5)*2*0.698
    fingers_[:, 4] = b[:,0]

    fingers_[:, 6] = b[:,1]
    fingers_[:, 7] = s[:,0]

    fingers_[:, 9] = b[:,2]
    fingers_[:, 10] = s[:,1]
    fingers_[:, 11] -= 0.5
    fingers_[:, 12] = b[:,3]
    fingers_[:, 13] = s[:,2]
    fingers_[:, 14:15]=(1-torch.rand((size, 1), device=device)**2)*0.785
    fingers_[:, 15] -= 0.5
    fingers_[:, 16] = b[:,4]
    fingers_[:, 17] = s[:,3]

    delta = torch.randn((size, 3), device=device)/2
    delta[:,0:2]/=5

    sampled_pose = torch.cat([alpha_,beta_,delta,gamma, fingers_], dim=1)

    return sampled_pose

def generate_random_Allergo_poses(size):

    alpha_ = torch.cat([random_unit_circle(size),torch.rand((size,1),device=device)*-1],dim=1)
    alpha_ = F.normalize(alpha_, dim=-1)

    beta_ = random_unit_circle(size)
    beta_ = F.normalize(beta_, dim=-1)

    fingers_ = torch.rand((size, 16), device=device)-0.5

    fingers_[:,1:2]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,2:3]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,3:4]=torch.rand((size, 1), device=device)

    fingers_[:,5:6]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,6:7]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,7:8]=torch.rand((size, 1), device=device)

    fingers_[:,9:10]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,10:11]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,11:12]=torch.rand((size, 1), device=device)

    '''thumb'''
    fingers_[:,12:13]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,13:14]=torch.rand((size, 1), device=device)
    fingers_[:,14:15]=1-torch.rand((size, 1), device=device)**2
    fingers_[:,15:16]=torch.rand((size, 1), device=device)


    delta = torch.randn((size, 3), device=device)/2
    delta[:,0:2]/=5

    sampled_pose = torch.cat([alpha_,beta_,delta, fingers_], dim=1)

    return sampled_pose

def generate_random_r_2f85_poses(size):

    alpha_ = torch.cat([random_unit_circle(size),torch.rand((size,1),device=device)*-1],dim=1)
    alpha_ = F.normalize(alpha_, dim=-1)

    beta_ = random_unit_circle(size)
    beta_ = F.normalize(beta_, dim=-1)


    delta = torch.randn((size, 3), device=device)/2
    delta[:,0:2]/=5

    sampled_pose = torch.cat([alpha_,beta_,delta], dim=1)

    return sampled_pose


if __name__ == "__main__":

    r=beta_peak_intensity_tensor(1000, 3, torch.tensor([0,-1,1]).to(device),[-2,2], peak_intensity=30.0)
    data=r[:,0].cpu().numpy()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()