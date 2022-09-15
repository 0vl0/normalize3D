import numpy as np
import matplotlib.pyplot as plt
from inertia import barycenter_field, get_ev_inertia, sort_eigen_vectors

def plot_cross_section(cross_section):
    """
    Plot a 2D cross-section
    Input:
        cross_section: 2D numpy array
    """
    Nx, Ny = cross_section.shape
    grid_x, grid_y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(grid_x, grid_y, c=cross_section, cmap="Reds", alpha=0.05)
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')
    ax.set_xlim(0,Nx)
    ax.set_ylim(0,Ny)

def plot_scalar_field_torch(scalar_field):
    """
    Plot batch of 3D torch scalar fields.
    Input:
        scalar_field: 5D torch Tensor
    """
    np_scalar_field = scalar_field.clone().detach().cpu().numpy()
    len_batch, _, Nx, Ny, Nz = np_scalar_field.shape
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    fig = plt.figure()
    for i in range(len_batch):
        ax = fig.add_subplot(len_batch+1,1,i+1, projection='3d')
        ax.scatter(grid_x, grid_y, grid_z, c=np_scalar_field[i][0], cmap="Reds", alpha=0.05)
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0,Nx)
    ax.set_ylim(0,Ny)
    ax.set_zlim(0,Nz)

def plot_scalar_field(scalar_field):
    """
    Plot 3D form represented by 3D scalar field.
    Plot is best rendered when scalar are in the range [0,1], consider
    normalizing if it's not the case.
    Input:
        scalar_field: 3D float numpy array
    """
    Nx, Ny, Nz = scalar_field.shape[0], scalar_field.shape[1], scalar_field.shape[2] 
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_x, grid_y, grid_z, c=scalar_field, cmap="Reds", alpha=0.05)
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0,Nx)
    ax.set_ylim(0,Ny)
    ax.set_zlim(0,Nz)
    plt.show(block=False)

def plot_inertia_numpy(np_tensor):
    """
    Plot 3D represented by 3D scalar field, 
    as well as the principal rotation axis.
    Inputs: 
        np_tensor: 3D float numpy array
    """
    Nx, Ny, Nz = np_tensor.shape[0], np_tensor.shape[1], np_tensor.shape[2] 
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    b = barycenter_field(np_tensor)
    eigen_values, Eigen_Vectors = get_ev_inertia(np_tensor, grid_x, grid_y, grid_z)
    EV = sort_eigen_vectors(eigen_values, Eigen_Vectors)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(grid_x, grid_y, grid_z, c=np_tensor, cmap="Reds", alpha=0.05)
    ax.quiver(b[0], b[1], b[2], EV[:,0], EV[:,1], EV[:,2], color=["red", "green", "gray"], length=40, alpha=1.)
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0,Nx)
    ax.set_ylim(0,Ny)
    ax.set_zlim(0,Nz)
    plt.show(block=False)

def plot_inertia_torch(scalar_field):
    """
    Plot batch of 3D scalar fields and the 3 principal rotation axis.
    Designed for scalar fields in the range [0,1], consider normalizing your scalar field before calling this function.
    Input: 
        scalar_field: 5D torch Tensor
    """
    np_tensor = scalar_field.clone().detach().cpu().numpy()
    len_batch, _, Nx, Ny, Nz = np_tensor.shape
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    
    fig = plt.figure()
    for i in range(len_batch):
        b = barycenter_field(np_tensor[i][0])
        eigen_values, Eigen_Vectors = get_ev_inertia(np_tensor[i][0], grid_x, grid_y, grid_z)
        EV = sort_eigen_vectors(eigen_values, Eigen_Vectors)
        ax = fig.add_subplot(len_batch, 1, i+1, projection='3d')
        ax.scatter(grid_x, grid_y, grid_z, c=np_tensor[i][0], cmap="Reds", alpha=0.05)
        ax.quiver(b[0], b[1], b[2], EV[:,0], EV[:,1], EV[:,2], color=["red", "green", "gray"], length=40, alpha=1.)
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0,Nx)
    ax.set_ylim(0,Ny)
    ax.set_zlim(0,Nz)