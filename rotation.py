import numpy as np 
import torch

def get_rotation_matrix(n, angle):
    """
    Return rotation matrix corresponding to normal vector n and angle angle.
    Inputs:
        n: 3x1 float =  normal vector
        angle: float, rotation angle in radians
    Output:
        3x3 float matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[c+(1-c)*n[0]**2, (1-c)*n[0]*n[1]+s*n[2], (1-c)*n[0]*n[2]-s*n[1]], 
                         [(1-c)*n[0]*n[1]-s*n[2], c+(1-c)*n[1]**2, (1-c)*n[1]*n[2]+s*n[0]],
                         [(1-c)*n[0]*n[2]+s*n[1], (1-c)*n[1]*n[2]-s*n[0], c+(1-c)*n[2]**2]])

def get_rotation_tensor(n, angle, device="cuda:0"):
    """
    Return rotation matrix as pytorch tensor corresponding to normal vector n and angle angle.
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    return torch.tensor([[c+(1-c)*n[0]**2, (1-c)*n[0]*n[1]+s*n[2], (1-c)*n[0]*n[2]-s*n[1]], 
                         [(1-c)*n[0]*n[1]-s*n[2], c+(1-c)*n[1]**2, (1-c)*n[1]*n[2]+s*n[0]],
                         [(1-c)*n[0]*n[2]+s*n[1], (1-c)*n[1]*n[2]-s*n[0], c+(1-c)*n[2]**2]]).to(device)

def rotate_matrix(points, m):
    """
    Apply rotation matrix m to list of points points.
    Computation is not vectorized, see rotate for parallel rotation computations.
    Inputs:
        points: float x float x float array
        m: 3x3 float matrix
    Output:
        float x float x float list
    """
    return list(map(lambda p: np.matmul(m,p), points))

def rotate_matrix_torch(points, m):
    return torch.stack(list(map(lambda p: torch.matmul(m, p), points)))

def rotate_matrix_reference(points, m, r):
    return torch.stack(list(map(lambda p: torch.matmul(m, p-r)+r, points)))

def rotate_points(points, n, angle):
    """
    Rotate all points around normal n with rotation angle angle
    Inputs:
        points: float x float x float array
        n: 3x1 float
        angle: float
    Output:
        float x float x float list
    """
    alpha_ij = get_rotation_matrix(n, angle)
    return list(map(lambda p: np.matmul(alpha_ij,p), points))

def rotate_points_torch(points, n, angle, device="cuda:0"):
    alpha_ij = get_rotation_tensor(n, angle, device)
    return list(map(lambda p: torch.matmul(alpha_ij,p), points))

def rotate_function_x(m,x,y,z,b,N):
    """
    Compute new x coordinates after rotation.
    See rotate(_torch) for a use case.
    Inputs:
        m: 3x3 float array = rotation matrix
        x,y,z: float array, intial coordinates
        b: float x float x float = center of rotation (reference by default)
        N: upper bound of x axis
    Output:
        new x coordinate
    """
    new_x = np.round(m[0,0]*(x-b[0]) + m[0,1]*(y-b[1]) + m[0,2]*(z-b[2]) + b[0]).astype(int)
    new_x[new_x == N-1] = N-2
    new_x[new_x == -1] = 0
    new_x[new_x < 0] = N-1
    new_x[new_x >= N] = N-1
    return new_x 

def rotate_function_y(m,x,y,z,b,N):
    """
    Compute new x coordinates after rotation.
    """
    new_y = np.round(m[1,0]*(x-b[0]) + m[1,1]*(y-b[1]) + m[1,2]*(z-b[2]) + b[1]).astype(int)
    new_y[new_y == N-1] = N-2
    new_y[new_y == -1] = 0
    new_y[new_y < 0] = N-1
    new_y[new_y >= N] = N-1
    return new_y 

def rotate_function_z(m,x,y,z,b,N):
    """
    apply rotation to z coordinate
    """
    new_z = np.round(m[2,0]*(x-b[0]) + m[2,1]*(y-b[1]) + m[2,2]*(z-b[2]) + b[2]).astype(int)
    new_z[new_z == N-1] = N-2
    new_z[new_z == -1] = 0
    new_z[new_z < 0] = N-1
    new_z[new_z >= N] = N-1
    return new_z 

def rotate_function_x_torch(m,x,y,z,b,N):
    new_x = torch.round(m[0,0]*(x-b[0]) + m[0,1]*(y-b[1]) + m[0,2]*(z-b[2]) + b[0]).int()
    new_x[new_x == N-1] = N-2
    new_x[new_x == -1] = 0
    new_x[new_x < 0] = N-1
    new_x[new_x >= N] = N-1
    return new_x 

def rotate_function_y_torch(m,x,y,z,b,N):
    """
    new y coordinate after rotation
    """
    new_y = torch.round(m[1,0]*(x-b[0]) + m[1,1]*(y-b[1]) + m[1,2]*(z-b[2]) + b[1]).int()
    new_y[new_y == N-1] = N-2
    new_y[new_y == -1] = 0
    new_y[new_y < 0] = N-1
    new_y[new_y >= N] = N-1
    return new_y 

def rotate_function_z_torch(m,x,y,z,b,N):
    """
    apply rotation to z coordinate
    """
    new_z = torch.round(m[2,0]*(x-b[0]) + m[2,1]*(y-b[1]) + m[2,2]*(z-b[2]) + b[2]).int()
    new_z[new_z == N-1] = N-2
    new_z[new_z == -1] = 0
    new_z[new_z < 0] = N-1
    new_z[new_z >= N] = N-1
    return new_z 

def rotate_torch(rotation_matrix_tab, field_tensor, references_tab, device="cuda:0", default_value = 0.):
    """
    Rotate 3D field with rotation matrix.
    Invert rotation matrix is computed, then new tensor field is indexed with corresponding values of the non rotated tensor.
    Limit conditions are as follow: 
        - new coordinates with negative values are 0 by default,
        - new coordinates with out-of-bounds values are sent to the edge of the new array 
        (with 1 more voxel than the orginial in each dimension), which contains a default value.
    Inputs:
        rotation_matrix_tab: m: ix3x3 float tensor = list of rotation matrices
        field_tensor: ixjxkxlxm float tensor
        reference: float x float x float tensor = reference point of the rotation
        device: String = device to which tensors are attached 
        default_value: float = values for out-of-bound rotated voxels
    Output:
        ixjxkxlxm float tensor = rotated scalar field
    """
    rotated_field = torch.zeros_like(field_tensor)
    len_batch = rotated_field.shape[0]

    s = field_tensor.shape
    # extra voxel in each coordinate will contain the default value for out-of-bound rotated voxels.
    bound_x, bound_y, bound_z = s[2]+1, s[3]+1, s[4]+1
    bounding_tensor_field = torch.ones((s[0], s[1], bound_x, bound_y, bound_z))*default_value
    bounding_tensor_field[:,:,:-1,:-1,:-1] = field_tensor

    grid_x, grid_y, grid_z = get_grids_torch(field_tensor)
    rotation_inverse_tab = [torch.transpose(rotation_matrix_tab[i],0,1) for i in range(s[0])]

    for i in range(len_batch):
        matrix_x = rotate_function_x_torch(rotation_inverse_tab[i], grid_x, grid_y, grid_z, references_tab[i], bound_x)
        matrix_y = rotate_function_y_torch(rotation_inverse_tab[i], grid_x, grid_y, grid_z, references_tab[i], bound_y)
        matrix_z = rotate_function_z_torch(rotation_inverse_tab[i], grid_x, grid_y, grid_z, references_tab[i], bound_z)
        rotated_field[i][0] = torch.tensor(bounding_tensor_field[i][0][matrix_x.long(), matrix_y.long(), matrix_z.long()])
    
    return rotated_field.cuda().float().to(device)

def rotate(rotation_matrix, field_tensor, reference, default_value=0.0):
    """
    Rotate 3D field with rotation matrix.
    Align rotation axes of 3D forms with orthonormal basis (x,y,z)
    """
    rotated_field = np.zeros_like(field_tensor)
    s = field_tensor.shape
    
    # extra voxel in each coordinate will contain the default value for out-of-bound rotated voxels.
    bound_x, bound_y, bound_z = s[0]+1, s[1]+1, s[2]+1
    bounding_field = np.ones((bound_x, bound_y, bound_z))*default_value
    bounding_field[:-1,:-1,:-1] = field_tensor

    grid_x, grid_y, grid_z = get_grids(field_tensor)
    rotation_inverse = rotation_matrix.transpose()

    matrix_x = rotate_function_x(rotation_inverse, grid_x, grid_y, grid_z, reference, bound_x)
    matrix_y = rotate_function_y(rotation_inverse, grid_x, grid_y, grid_z, reference, bound_y)
    matrix_z = rotate_function_z(rotation_inverse, grid_x, grid_y, grid_z, reference, bound_z)
    rotated_field = np.array(bounding_field[matrix_x.astype(int), matrix_y.astype(int), matrix_z.astype(int)])
    return rotated_field

def get_grids_torch(scalar_field, d="cuda:0"):
    """
    Return meshgrid of coordinates for vectorization.
    Inputs:
        scalar_field: batch of 3D images, 5D torch Tensor
        d: String, device to which tensors are attached
    Outputs:
        grid_x: kxlxm int tensor = x coordinates 
        grid_y: kxlxm int tensor = y coordinates 
        grid_z: kxlxm int tensor = z coordinates
    """
    s = scalar_field.shape
    Nx, Ny, Nz = s[2], s[3], s[4]
    grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(Nx), torch.arange(Ny), torch.arange(Nz), indexing="ij")
    return grid_x.to(d), grid_y.to(d), grid_z.to(d)

def get_grids(scalar_field):
    """
    Return meshgrid of coordinates for vectorization.
    Inputs:
        scalar_field: 3D float numpy array
    Outputs:
        grid_x, grid_y, grid_z: meshgrid of 3D space
    """
    s = scalar_field.shape
    Nx, Ny, Nz = s[0], s[1], s[2]
    return np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")

def reflection_reference(list_points_xyz, reference):
    """
    Brute force implementation of reflection around reference point.
    Inputs:
        list_points_xyz: int array x int array x int array = list of x-y-z coordinates
        reference: 3x1 float array
    Output: int array x int array x int array = reflected list of x-y-z coordinates
    """
    x,y,z = np.array(list_points_xyz[0])-reference[0], np.array(list_points_xyz[1])-reference[1], np.array(list_points_xyz[2])-reference[2]
    if abs(np.min(x)) > np.max(x):
        x *= -1
    if abs(np.min(y)) > np.max(y):
        y *= -1
    if abs(np.min(z)) > np.max(z):
        z *= -1
    x += reference[0]
    y += reference[1]
    z += reference[2]
    x[x<0] =  0
    x[x>len(x)-1] = len(x)-1
    y[y<0] = 0
    y[y>len(y)-1] = len(y)-1
    z[z<0] = 0
    z[z>len(z)-1] = len(z)-1
    return (x,y,z)

def reflection_tensor(form_tensor, reference):
    """
    Brute force implementeation of reflection around reference
    Inputs:
        form_tensor: 3D numpy array
        reference: int x int x int numpy array, coordinates of reference point 
    Output:
        Reflected 3D numpy array
    """
    Nx, Ny, Nz = form_tensor.shape
    list_X_Y_Z = np.where(form_tensor==1)
    list_X_Y_Z_reflected = reflection_reference(list_X_Y_Z, reference=reference)
    return convert_X_Y_Z_to_matrix(list_X_Y_Z_reflected, Nx, Ny, Nz)

def convert_X_Y_Z_to_matrix(list_X_Y_Z, Nx, Ny, Nz):
    """
    Convert tuple of list of points into 3D numpy array
    """
    A = np.zeros((Nx, Ny, Nz))
    A[np.round(list_X_Y_Z[0]).astype(int), np.round(list_X_Y_Z[1]).astype(int), np.round(list_X_Y_Z[2]).astype(int)] = 1
    return A

def reflection_tensor_torch(form_tensor, references):
    """
    Brute force method to make reflection on tensors
    Inputs:
        form_tensor: 5D pytorch tensor
        references: ix3 pytorch tensor: references of reflections
    Ouput:
        Reflected 5D pytorch tensor
    """
    len_batch, _, Nx, Ny, Nz = form_tensor.shape
    reflected_tensor = torch.zeros_like(form_tensor)
    for i in range(len_batch):
        list_X_Y_Z = torch.where(form_tensor[i][0]==1)
        list_X_Y_Z_reflected = reflection_reference_torch(list_X_Y_Z, reference=references[i])
        reflected_tensor[i][0] = convert_X_Y_Z_to_tensor(list_X_Y_Z_reflected, Nx, Ny, Nz)
    return reflected_tensor

def reflection_reference_torch(list_points_xyz, reference):
    """
    Brute force implementation of reflection around reference point.
    Inputs:
        list_points_xyz: int array x int array x int array = list of x-y-z coordinates
        reference: 3x1 float array
    Output: int array x int array x int array = reflected list of x-y-z coordinates
    """
    x,y,z = torch.tensor(list_points_xyz[0])-reference[0], torch.tensor(list_points_xyz[1])-reference[1], torch.tensor(list_points_xyz[2])-reference[2]
    if abs(torch.min(x)) > torch.max(x):
        x *= -1
    if abs(torch.min(y)) > torch.max(y):
        y *= -1
    if abs(torch.min(z)) > torch.max(z):
        z *= -1
    x += reference[0]
    y += reference[1]
    z += reference[2]
    x[x<0] =  0
    x[x>len(x)-1] = len(x)-1
    y[y<0] = 0
    y[y>len(y)-1] = len(y)-1
    z[z<0] = 0
    z[z>len(z)-1] = len(z)-1
    return (x,y,z)

def convert_X_Y_Z_to_tensor(list_X_Y_Z, Nx, Ny, Nz):
    """
    Convert tuple of list of points into 3D numpy array
    """
    A = torch.zeros((Nx, Ny, Nz))
    A[torch.round(list_X_Y_Z[0]).long(), torch.round(list_X_Y_Z[1]).long(), torch.round(list_X_Y_Z[2]).long()] = 1
    return A

if __name__ == "__main__":
    print(help(reflection_tensor))
