import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_3d_gradients(grid_points, gradients, radius, sphere_radius=1, cell=False):
    x, y, z = grid_points
    grad_x, grad_y, grad_z = gradients
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    if cell:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = sphere_radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = sphere_radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        for i, j, k in itertools.product(range(x.shape[0]), range(x.shape[1]), range(x.shape[2])):
            xi, yi, zi = x[i, j, k], y[i, j, k], z[i, j, k]
            r = np.sqrt(xi**2 + yi**2 + zi**2)
            if r <= radius:
                ax.plot_wireframe(sphere_x + xi, sphere_y + yi, sphere_z + zi, color='b', alpha=0.1)
    

    ax.quiver(x, y, z, grad_x, grad_y, grad_z, length=sphere_radius, normalize=True, color='orange')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax

