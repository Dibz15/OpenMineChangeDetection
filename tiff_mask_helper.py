from gimpfu import *
import os
import glob


def open_new_image(file_path):
    if os.path.isfile(file_path):
        image = pdb.file_tiff_load(file_path, file_path)
        display = pdb.gimp_display_new(image)
        return image
    else:
        return None
    
def add_as_layer(image, file_path, layer_name, opacity=100.0):
    if image is None:
        return
    # If this is not the first image, open the file as a new layer
    layer = pdb.gimp_file_load_layer(image, file_path)
    layer.name = layer_name
    layer.opacity = opacity
    # Add the layer to the image
    image.add_layer(layer, -1)

def find_image_path(dir, file_suffix):
    file_pattern = os.path.join(dir, "*" + file_suffix + ".tif")
    # Use glob to find files that match the pattern
    matching_files = glob.glob(file_pattern)
    return matching_files[0]

def open_tif_files(root_dir, file_suffix):
    # List all subdirectories in the root directory
    # subdirs = ['CVA', 'PCA', 'imageA', 'imageB']

    # image = None

    cva_path = os.path.join(root_dir, 'CVA')
    pca_path = os.path.join(root_dir, 'PCA')
    imageA_path = os.path.join(root_dir, 'imageA')
    imageB_path = os.path.join(root_dir, 'imageB')

    cva_image = open_new_image(find_image_path(cva_path, file_suffix))
    pca_image = open_new_image(find_image_path(pca_path, file_suffix))
    imageB = open_new_image(find_image_path(imageB_path, file_suffix))
    add_as_layer(imageB, find_image_path(imageA_path, file_suffix), "imageA")
    add_as_layer(imageB, find_image_path(pca_path, file_suffix), "PCA", opacity=25.0)
    add_as_layer(imageB, find_image_path(cva_path, file_suffix), "CVA", opacity=25.0)

    # # For each subdirectory
    # for subdir in subdirs:
    #     # Construct the file pattern
    #     file_pattern = os.path.join(root_dir, subdir, "*" + file_suffix + ".tif")

    #     # Use glob to find files that match the pattern
    #     matching_files = glob.glob(file_pattern)

    #     # For each matching file
    #     for file_path in matching_files:
    #         # Check if the file exists
    #         if os.path.isfile(file_path):
    #             # If this is the first image, open it normally
    #             if image is None:
    #                 image = pdb.file_tiff_load(file_path, file_path)
    #                 display = pdb.gimp_display_new(image)
    #             else:
    #                 # If this is not the first image, open the file as a new layer
    #                 layer = pdb.gimp_file_load_layer(image, file_path)

    #                 # Add the layer to the image
    #                 image.add_layer(layer, -1)

register(
    "python_fu_tiff_mask_helper",
    "Open TIF Files",
    "Opens a specific .tif file from each subdirectory of a given root directory and adds them as layers to the same image",
    "Your Name",
    "Your Name",
    "2023",
    "<Toolbox>/Xtns/Languages/Python-Fu/_TIFF Mask Helper...",
    "",
    [
        (PF_DIRNAME, "root_dir", "Root directory", "/"),
        (PF_STRING, "file_suffix", "File suffix", "0001"),
    ],
    [],
    open_tif_files)

main()
