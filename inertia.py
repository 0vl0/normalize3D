import numpy as np
import torch
from rotation import get_rotation_tensor, reflection_tensor_torch, rotate_points_torch, get_grids_torch, get_grids, get_rotation_matrix, rotate, rotate_points, rotate_torch, reflection_tensor

def barycenter_voxel(list_coordinates):
    """
    Compute barycenter of voxellized (0-1) 3D form.
    See barycenter_field to compute barycenter of 3D scalar field.
    """
    length = len(list_coordinates[0])
    sum_x = np.sum(list_coordinates[0])
    sum_y = np.sum(list_coordinates[1])
    sum_z = np.sum(list_coordinates[2])
    return sum_x/length, sum_y/length, sum_z/length

def barycenter_field(scalar_field):
    """
    Compute barycenter of 3D  scalar field.
    Input: 
        scalar_field, 3D numpy array
    Output: 
        3D float numpy array
    """
    grid_x, grid_y, grid_z = get_grids(scalar_field)
    b_x = np.sum(grid_x*scalar_field)/np.sum(scalar_field)
    b_y = np.sum(grid_y*scalar_field)/np.sum(scalar_field)
    b_z = np.sum(grid_z*scalar_field)/np.sum(scalar_field)
    return np.array([b_x, b_y, b_z])

def barycenter_field_torch(scalar_field, grid_x, grid_y, grid_z, d="cuda:0"):
    """
    Compute barycenter of scalar distribution on 3D space
    Inputs: scalar_field, float tensor
            grid_x, grid_y, grid_z: meshgrid of 3D space
            d: gpu to which barycenter is attached
    Output : intxintxint Tensor
    """
    #grid_x, grid_y, grid_z = get_grids_torch(scalar_field)
    b_x = torch.sum(grid_x.to(d)*scalar_field)/torch.sum(scalar_field)
    b_y = torch.sum(grid_y.to(d)*scalar_field)/torch.sum(scalar_field)
    b_z = torch.sum(grid_z.to(d)*scalar_field)/torch.sum(scalar_field)
    return torch.tensor([b_x, b_y, b_z])

def matrix_inertia_field(scalar_field, reference, grid_x, grid_y, grid_z):
    """
    Compute matrix of inertia of 3D image according to reference point
    Inputs: 
        scalar_tensor: 3D numpy float
        reference: 3x1 float
        grid_x, grid_y, grid_z: meshgrid of 3D space
    Output: 3x3 float array
    """
    gx = (grid_x - reference[0]) * scalar_field
    gy = (grid_y - reference[1]) * scalar_field
    gz = (grid_z - reference[2]) * scalar_field
    Ix = np.sum((gy**2+gz**2)*scalar_field)
    Iy = np.sum((gx**2+gz**2)*scalar_field)
    Iz = np.sum((gx**2+gy**2)*scalar_field)
    Ixy = -np.sum((gx*gy)*scalar_field)
    Ixz = -np.sum((gx*gz)*scalar_field)
    Iyz = -np.sum((gy*gz)*scalar_field)
    return np.array([[Ix, Ixy, Ixz], [Ixy, Iy, Iyz], [Ixz, Iyz, Iz]])

def matrix_inertia_field_torch(tensor_proba, barycenter, grid_x, grid_y, grid_z):
    """
    See matrix_inertia_field documentation
    """
    gx = (grid_x - barycenter[0]) 
    gy = (grid_y - barycenter[1]) 
    gz = (grid_z - barycenter[2])
    Ix = torch.sum((gy**2+gz**2)*tensor_proba)
    Iy = torch.sum((gx**2+gz**2)*tensor_proba)
    Iz = torch.sum((gx**2+gy**2)*tensor_proba)
    Ixy = -torch.sum((gx*gy)*tensor_proba)
    Ixz = -torch.sum((gx*gz)*tensor_proba)
    Iyz = -torch.sum((gy*gz)*tensor_proba)
    return torch.tensor([[Ix, Ixy, Ixz], [Ixy, Iy, Iyz], [Ixz, Iyz, Iz]])

def get_ev_inertia(scalar_field, grid_x, grid_y, grid_z):
    """
    Return eigen vectors and eigen values of inertia matrix (calculated around barycenter) 
    of 3D scalar field tensor_probas
    Inputs:
        scalar_field: 3D float numpy arary
        grid_x, grid_y, grid_z: meshgrid of 3D space
    """
    b = barycenter_field(scalar_field)
    m = matrix_inertia_field(scalar_field, b, grid_x, grid_y, grid_z)
    eigen_values, Eigen_Vectors = np.linalg.eig(m)
    return eigen_values, Eigen_Vectors

def sort_eigen_vectors(eigen_values, eigen_vectors):
    """
    Sort eigen vectors according to eigen values, descending order.
    Inputs:
        eigen values: 3x1 float array
        eigen vectors: (3x1 float array)^3
    Ouput:
        Sorted eigen vectors
    """
    indices = np.argsort(eigen_values)[::-1]
    EV1 = np.transpose(eigen_vectors)[indices]
    return np.real(EV1)

def sort_eigen_vectors_torch(eigen_values, eigen_vectors):
    """
    See sort_eigen_vectors documentation.
    """
    _, indices = torch.sort(eigen_values, descending=True)
    EV1 = torch.transpose(eigen_vectors, 0, 1)[indices]
    return torch.real(EV1)

def get_rotation_params_torch(scalar_field, grid_x, grid_y, grid_z, device="cuda:0"):
    """
    Get rotation params to align rotation axes of batch of 3D images with orthonormal basis (x,y,z)
    Inputs: 
        scalar_field: 3D float Tensor - shape = jxkxl
        grid_x, grid_y, grid_z: meshgrid of 3D space for vectorization
    Outputs: 
        rotation_matrix: 3x3 float tensor
        barycenter: 3x1 float Tensor
    """
    b = barycenter_field_torch(scalar_field, grid_x, grid_y, grid_z, device)
    
    # *** alignement of eigen vectors with biggest eigen value with x axis ***    
    mi1 = matrix_inertia_field_torch(scalar_field, b, grid_x, grid_y, grid_z)
    eigen_values1, Eigen_Vectors1 = torch.eig(mi1, eigenvectors=True)
    eigen_values1 = abs(eigen_values1[:,0])
    # sorting the eigen vectors according to absolute value of their eigen values
    sorted_EV1 = sort_eigen_vectors_torch(eigen_values1, Eigen_Vectors1)
    # saving the second eigen vector for the second rotation
    EV2 = sorted_EV1[1].to(device)
    # computing axis of rotation and angle
    normalx = torch.cross(torch.tensor([1.,0.,0.]),sorted_EV1[0])
    normalx /= torch.norm(normalx)
    # computing rotation matrix from Rowenhorst's article
    angle1 = torch.acos(torch.dot(sorted_EV1[0],torch.tensor([1.,0.,0.])))
    rotation_matrix_1 = get_rotation_tensor(normalx, angle1, device)
    # rotating second eigen vector
    EV2_rotated = rotate_points_torch([EV2], normalx, angle1, device)

    # *** alignement of eigen vectors with second biggest eigen value with y axis ***
    vector_y = torch.tensor([0.,1.,0.]).to(device)
    normaly = torch.cross(vector_y,EV2_rotated[0])
    normaly /= torch.norm(normaly)
    angle2 = torch.acos(torch.dot(EV2_rotated[0],vector_y))
    rotation_matrix_2 = get_rotation_tensor(normaly, angle2, device)

    return b, torch.matmul(rotation_matrix_2, rotation_matrix_1)

def normalize_orientation(field):
    """
    Normalized orientation of 3D field
    Input:
        field: 3D numpy array
    Output:
        Orientation normalized 3D numpy array
    """
    barycenter, rotation_matrix = get_rotation_params(field)
    rotated_field = rotate(rotation_matrix, field, barycenter)
    return reflection_tensor(rotated_field, barycenter_field(rotated_field))

def normalize_orientation_torch(field):
    """
    Normalize orientation of 3D field
    Inputs:
        field: batch of 3D fields, 5D torch tensor
    Output:
        Orientation normalized batch of 3D fields
    """
    grid_x, grid_y, grid_z = get_grids_torch(field)
    rotation_params = [get_rotation_params_torch(field[i][0], grid_x, grid_y, grid_z) for i in range(field.shape[0])]
    barycenter_tab, rotation_tensor_tab = list(zip(*rotation_params))
    rotated_field = rotate_torch(rotation_tensor_tab, field, barycenter_tab)
    references = [barycenter_field_torch(rotated_field[i][0], grid_x, grid_y, grid_z) for i in range(rotated_field.shape[0])]
    return reflection_tensor_torch(rotated_field, references)

def get_rotation_params(scalar_field):
    """
    Get rotation params to align rotation axes of 3D image with orthonormal basis (x,y,z)
    Inputs: 
        scalar_field: 3D numpy array
    Outputs: 
        barycenter: 3x1 float tensor
        rotation_matrix: 3x3 float tensor
    """
    grid_x, grid_y, grid_z = get_grids(scalar_field)
    # initial barycenter
    b = barycenter_field(scalar_field)
    
    # *** alignement of eigen vectors with biggest eigen value with x axis ***
    mi1 = matrix_inertia_field(scalar_field, b, grid_x, grid_y, grid_z)
    eigen_values1, Eigen_Vectors1 = np.linalg.eig(mi1)
    eigen_values1 = abs(eigen_values1)
    # sorting the eigen vectors according to absolute value of their eigen values
    sorted_EV1 = sort_eigen_vectors(eigen_values1, Eigen_Vectors1)
    # saving the second eigen vector for the second rotation
    EV2 = sorted_EV1[1]
    # computing axis of rotation and angle
    normalx = np.cross(np.array([1.,0.,0.]),sorted_EV1[0])
    normalx /= np.linalg.norm(normalx)
    # computing rotation matrix from Rowenhorst's article
    angle1 = np.arccos(np.dot(sorted_EV1[0],np.array([1.,0.,0.])))
    rotation_matrix_1 = get_rotation_matrix(normalx, angle1)
    # rotating second eigen vector
    EV2_rotated = rotate_points([EV2], normalx, angle1)

    # *** alignement of eigen vectors with second biggest eigen value with y axis ***
    vector_y = np.array([0.,1.,0.])
    normaly = np.cross(vector_y,EV2_rotated[0])
    normaly /= np.linalg.norm(normaly)
    angle2 = np.arccos(np.dot(EV2_rotated[0],vector_y))
    rotation_matrix_2 = get_rotation_matrix(normaly, angle2)

    return b, np.matmul(rotation_matrix_2, rotation_matrix_1)