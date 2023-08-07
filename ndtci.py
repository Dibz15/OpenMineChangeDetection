import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calculate_average_mask(directory, facility_name='', ignore=[], prefix='pred_'):
    # Find all GeoTIFF files in the directory
    mask_files = glob.glob(os.path.join(directory, f'{prefix}{facility_name}*.tif'))

    masks = []

    # Read each mask file
    for mask_file in mask_files:
        if mask_file.endswith('_NDTCI.tif'):
            continue
        
        on_ignore_list = False
        for i in ignore:
            if i in mask_file:
                on_ignore_list = True
                break
        if on_ignore_list:
            continue
        with rasterio.open(mask_file) as src:
            mask = src.read(1)  # Reading the first band
            mask = np.clip(mask, a_min=0, a_max=1).astype(float)
            masks.append(mask)
    
    # Convert the list of masks to a 3D numpy array and calculate the average
    average_mask = np.mean(masks, axis=0)
    
    return average_mask

def calculate_x_stable(average_mask, quantile=0.25):
    # Flatten the average mask to a 1D array
    flat_mask = average_mask.flatten()

    # Calculate the value at the given quantile
    x_stable = np.quantile(flat_mask, quantile)

    return x_stable

def calculate_NDTCI(input_filepath, x_stable, eps=1e-6, area_mask=None):
    # Open the GeoTIFF mask file
    with rasterio.open(input_filepath) as src:
        x_activity = src.read(1)  # Reading the first band
        x_activity = np.clip(x_activity, a_min=0, a_max=1).astype(float)
        if area_mask is not None:
            x_activity = (x_activity.astype(np.uint8) & area_mask.astype(np.uint8)).astype(float)
    # Compute the NDTCI
    NDTCI = (x_activity - x_stable) / (x_activity + x_stable + eps)
    
    # Normalize the NDTCI to [0, 1] for coloring
    # NDTCI_normalized = (NDTCI + 1) / 2    
    # Define a colormap to go from red/brown (high values) to blue/green (low values)
    colormap = plt.get_cmap('RdYlBu_r')

    # Apply the colormap to the NDTCI matrix (this will create a 3D array)
    NDTCI_color = colormap(NDTCI)[:,:,:3] # Cut out alpha channel from colormap result
    # Get the original metadata from the source file
    meta = src.meta
    meta.update(count=3, dtype=rasterio.uint8)
    
    return NDTCI, NDTCI_color, meta

def write_average_NDTCI(NDTCI_matrices, output_filepath, meta):
    if len(NDTCI_matrices) == 0:
        print('No matrices given to calculate average NDTCI')
        return
    # Compute the average NDTCI matrix
    average_NDTCI = np.mean(NDTCI_matrices, axis=0)

    # Apply the colormap to the average NDTCI matrix (this will create a 3D array)
    colormap = plt.get_cmap('RdYlBu_r')
    average_NDTCI_color = colormap(average_NDTCI)[:,:,:3] # cut out alpha channel from colormap result

    # Write the average NDTCI GeoTIFF
    with rasterio.open(output_filepath, 'w', **meta) as dst:
        for channel in range(average_NDTCI_color.shape[2]):
            chc = average_NDTCI_color[:, :, channel]
            dst.write((chc * 255).astype(np.uint8), channel + 1)

    # Plotting the average NDTCI matrix
    fig, ax = plt.subplots()

    # Display the average NDTCI matrix using the colormap
    img = ax.imshow(average_NDTCI, cmap=colormap)
    # Create an axes for the colorbar
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create a colorbar
    cbar = fig.colorbar(img, cax=cax)

    # Save the plot as a .png file
    plt.savefig(output_filepath.replace('.tif', '.png'))
    plt.close()

