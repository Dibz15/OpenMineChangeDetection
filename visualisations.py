"""
Author: Austin Dibble

This file falls under the repository's OSL 3.0.

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from tiler import Tiler, Merger
import pandas as pd
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
import os
from torchgeo.transforms import AppendNDVI, AppendNDWI
from .ndtci import calculate_average_mask, calculate_x_stable, calculate_NDTCI, write_average_NDTCI
import seaborn as sns
from scipy import stats

def predict_file_mask(model, dataset, pred_func, device, file_index, threshold=None, plot=False, output_path=None):
    """
    Predict the mask for a specific file using the provided model and dataset.

    Parameters:
        model (torch.nn.Module): The PyTorch model used for prediction.
        dataset (torch.utils.data.Dataset): The dataset containing the image tiles for prediction.
        pred_func (function): The prediction function that takes the model, input data, and device and returns predictions.
        device (str or torch.device): The device to use for prediction (e.g., 'cuda' or 'cpu').
        file_index (int): The index of the file in the dataset to predict the mask for.
        threshold (float, optional): The threshold value to apply to the predicted mask.
                                     If provided, the mask will be binarized. Defaults to None.
        plot (bool, optional): Whether to plot and display the predicted mask. Defaults to False.
        output_path (str, optional): The path to save the predicted mask as a GeoTIFF file.
                                     If not provided, the mask will not be saved. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            - final_mask (numpy.ndarray): The predicted mask as a 2D numpy array.
            - image_area (int): The area of the original full image.

    Example:
        # Assuming the model, dataset, pred_func, and device are defined.
        file_index = 0
        threshold_value = 0.5
        prediction, image_area = predict_file_mask(model, dataset, pred_func, 'cuda', file_index, threshold=threshold_value, plot=True, output_path='/path/to/output_mask.tif')
    """
    batch_size = 3 # Set your batch size
    image_tile_map = dataset.get_image_tile_ranges()
    chip_idx_range = image_tile_map[file_index]
    tiler, full_image_shape = dataset._get_tiler(file_index, channels_override = 1)

    full_meta = dataset.get_tile_metadata(chip_idx_range[0], mask = True, no_tile = True)
    merger_out = Merger(tiler)

    tile_batch = []
    tile_ids = []
    for absolute_chip_id in range(chip_idx_range[0], chip_idx_range[1]+1):
        relative_tile_id = absolute_chip_id - chip_idx_range[0]
        tile_dict = dataset[absolute_chip_id]
        tile_batch.append(tile_dict)
        tile_ids.append(relative_tile_id)

        # Check if batch is full or if we are at the end of the loop
        if len(tile_batch) == batch_size or absolute_chip_id == chip_idx_range[1]:
            batch_input = torch.stack([tile['image'] for tile in tile_batch]).to(device)

            mask_pred_batch = pred_func(model, {'image': batch_input}, device)
            for i, mask_pred in enumerate(mask_pred_batch):
                mask_pred_numpy = mask_pred.unsqueeze(0).detach().cpu().numpy()
                merger_out.add(tile_ids[i], mask_pred_numpy)

            # Reset the batch
            tile_batch = []
            tile_ids = []

    final_mask = merger_out.merge(unpad=True).squeeze(0)
    if threshold is not None:
        final_mask = final_mask > threshold
    if plot:
        plt.figure()
        plt.imshow(final_mask)
        plt.show()

    if output_path is not None:
        with rasterio.open(output_path, 'w', **full_meta) as dst:
            dst.write((final_mask * 255).astype(np.uint8), 1)

    return final_mask, (full_image_shape[1]*full_image_shape[2])

def create_timeline_mask(date_list, mask_list, date_range_list, path=None):
    """
    Create a timeline mask visualizing multiple masks with associated dates. Saves the created plot to the given path.

    Parameters:
        date_list (list): A list of dates corresponding to each mask in mask_list.
                          Each date should be a string in a format recognized by pandas.to_datetime.
        mask_list (list): A list of masks, where each mask is a 2D numpy array.
                          The masks should have the same shape and be in binary format (0 or 1).
        date_range_list (list): A list of date ranges corresponding to each mask in mask_list.
                                Each date range can be a string representing a period or any relevant label.
        path (str, optional): The path to save the generated timeline mask as an image.
                              If not provided, the plot will be displayed but not saved. Defaults to None.

    Returns:
        None

    Example:
        date_list = ['2023-01-01', '2023-02-01', '2023-03-01']
        mask_list = [mask_1, mask_2, mask_3]
        date_range_list = ['2023-01-01 - 2023-02-01', '2023-02-01 - 2023-03-01', 'Mar 2023']
        create_timeline_mask(date_list, mask_list, date_range_list, path='/path/to/timeline_mask.png')
    """
    # Normalize dates to the range [0, 1] for color mapping
    min_date, max_date = pd.to_datetime(min(date_list)), pd.to_datetime(max(date_list))
    if min_date != max_date:
        norm_dates = [(pd.to_datetime(d) - min_date) / (max_date - min_date) for d in date_list]
    else:
        norm_dates = [0.5] * len(date_list)

    cmap = plt.get_cmap('viridis')
    hybrid_mask = np.zeros((*mask_list[0].shape, 3))  # Initialize with zeros, expecting 3D data for RGB
    for mask, norm_date in zip(mask_list, norm_dates):
        # Replace hybrid_mask with color-coded mask where mask == 1
        hybrid_mask[mask == 1] = np.array(cmap(norm_date))[:3]  # Take RGB, ignore alpha

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(hybrid_mask, alpha=0.6)

    # Create legend
    custom_lines = [Line2D([0], [0], color=cmap(i / (len(date_list)-1)), lw=4) for i in range(len(date_list))]
    ax.legend(custom_lines, [str(d) for d in sorted(date_range_list)], title='Dates', loc='upper right')

    if path is not None:
        # Save the figure for this facility
        plt.savefig(path, dpi=300)
    else:
        plt.show()
    plt.close()

def predict_masks(model, dataset, pred_func, device, threshold=None, output_dir=None, filter_area=False, facilities=None):
    """
    Predict masks for multiple facilities using the provided model and dataset.

    Parameters:
        model (torch.nn.Module): The PyTorch model used for prediction.
        dataset (torch.utils.data.Dataset): The dataset containing image tiles for prediction.
        pred_func (function): The prediction function that takes the model, input data, and device and returns predictions.
        device (str or torch.device): The device to use for prediction (e.g., 'cuda' or 'cpu').
        threshold (float, optional): The threshold value to apply to the predicted masks for binarization. Defaults to None.
        output_dir (str, optional): The directory to save the predicted masks and the timeline mask images.
                                    If not provided, the images will not be saved. Defaults to None.
        filter_area (bool, optional): Whether to filter the predicted masks using area masks. Defaults to False.
        facilities (list, optional): A list of facility names to process. If None, all facilities in the dataset will be used.

    Returns:
        pandas.DataFrame: A DataFrame containing information about the predicted masks and area changes for each facility.

    Example:
        # Assuming the model, dataset, and pred_func are defined.
        facilities_to_process = ['facility_A', 'facility_B']
        output_directory = '/path/to/output_directory'
        predicted_changes_df = predict_masks(model, dataset, pred_func, torch.device('cuda'), threshold=0.5, output_dir=output_directory, filter_area=True, facilities=facilities_to_process)
    """

    change_list = []
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    if facilities is None:
        facilities = dataset.get_facilities()

    for facility in tqdm(sorted(facilities)):
        facility_file_indices = dataset.get_facility_file_indices(facility)
        mask_list = []
        date_list = []
        cumulative_mask = None  # Initialize the cumulative mask

        if filter_area:
            area_mask_path = os.path.join(dataset.root_dir, 'area_mask', f'{facility}.tif')
            if os.path.isfile(area_mask_path):
                area_mask, _ = dataset._load_image(area_mask_path)
                area_mask = np.clip(area_mask, a_min=0, a_max=1).astype(np.uint8)
                area_mask = area_mask.squeeze(0)

        date_range_list = []  # This will store the date ranges for each file index
        for file_index in facility_file_indices:
            file_info = dataset.file_list[file_index]
            predate, postdate = file_info[-3:-2]
            date_range_list.append(f"{predate} - {postdate}")

            if output_dir is not None:
                output_path = os.path.join(output_dir, f'pred_{facility}_{file_index}_{predate}_{postdate}.tif')
            else:
                output_path = None

            mask, image_pixels = predict_file_mask(model, dataset, pred_func, device, file_index, threshold, plot=False, output_path=output_path)
            if filter_area:
                mask = mask & area_mask

            if cumulative_mask is None:
                cumulative_mask = mask  # Start with the first mask
            else:
                cumulative_mask = np.logical_or(cumulative_mask, mask)  # Logical OR operation

            cumulative_area_change = cumulative_mask.sum() * 100  # Total unique area change

            area_change = mask.sum() * 100  # (each pixel is 100 m^2)
            image_area = image_pixels * 100  # each pixel is 100m^2
            prop = float(area_change) / float(image_area)
            cumu_prop = float(cumulative_area_change) / float(image_area)
            # Append the mask and dates to the respective lists
            mask_list.append(mask)
            date_list.append(postdate)  # We only need postdate for coloring

            change_list.append({"facility": facility, "index": file_index,
                                "predate": predate, "postdate": postdate,
                                "change_area": area_change, "image_area": image_area,
                                "proportion": prop, "cumulative_area": cumulative_area_change,
                                "cumulative_prop": cumu_prop})

        if output_dir is not None:
            create_timeline_mask(date_list, mask_list, date_range_list,
                                path=os.path.join(output_dir, f'{facility}_hybrid_mask.png'))

    return pd.DataFrame(change_list)

def predict_index(dataset, index_func, device, file_index, plot=False, output_path=None):
    """
    Predict an index for a specific file using the provided index function and dataset.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing image tiles for prediction.
        index_func (function): The index function that takes a tile dictionary and device and returns the index values.
        device (str or torch.device): The device to use for prediction (e.g., 'cuda' or 'cpu').
        file_index (int): The index of the file in the dataset to predict the index for.
        plot (bool, optional): Whether to plot and display the predicted indices for both images. Defaults to False.
        output_path (str, optional): The path to save the predicted index as a GeoTIFF file.
                                     If not provided, the index will not be saved. Defaults to None.

    Returns:
        tuple: A tuple containing three elements:
            - final_index_pre (numpy.ndarray): The predicted index for the first image as a 2D numpy array.
            - final_index_post (numpy.ndarray): The predicted index for the second image as a 2D numpy array.
            - image_area (int): The area of the original full image.

    Example:
        # Assuming the dataset and index_func are defined.
        file_index = 0
        index_output, _, image_area = predict_index(dataset, index_func, 'cuda', file_index, plot=True, output_path='/path/to/index_output.tif')
    """
    image_tile_map = dataset.get_image_tile_ranges()
    chip_idx_range = image_tile_map[file_index]
    tiler, full_image_shape = dataset._get_tiler(file_index, channels_override = 1)
    full_meta = dataset.get_tile_metadata(chip_idx_range[0], mask = True, no_tile = True)
    merger_out_pre = Merger(tiler)
    merger_out_post = Merger(tiler)

    for absolute_chip_id in range(chip_idx_range[0], chip_idx_range[1]+1):
        relative_tile_id = absolute_chip_id - chip_idx_range[0]
        tile_dict = dataset[absolute_chip_id]

        pre_index, post_index = index_func(tile_dict, device)
        pre_index_numpy = pre_index.detach().cpu().numpy()
        post_index_numpy = post_index.detach().cpu().numpy()
        merger_out_pre.add(relative_tile_id, pre_index_numpy)
        merger_out_post.add(relative_tile_id, post_index_numpy)

    final_index_pre = merger_out_pre.merge(unpad=True).squeeze(0)
    final_index_post = merger_out_post.merge(unpad=True).squeeze(0)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        img1 = axs[0].imshow(final_index_pre, cmap='RdYlGn', vmin=0, vmax=1)
        axs[0].set_title('Index Image 1')
        img2 = axs[1].imshow(final_index_post, cmap='RdYlGn', vmin=0, vmax=1)
        axs[1].set_title('Index Image 2')
        cbar = fig.colorbar(img1, ax=axs.ravel().tolist(), fraction=.1)
        cbar.set_label('Index')
        plt.show()

    if output_path is not None:
        with rasterio.open(output_path, 'w', **full_meta) as dst:
            dst.write((final_index_pre * 255).astype(np.uint8), 1)

    return final_index_pre, final_index_post, (full_image_shape[1]*full_image_shape[2])

def predict_indices(dataset, pred_func, device):
    """
    Predict indices for all files in the dataset using the provided index prediction function.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing image tiles for prediction.
        pred_func (function): The index prediction function that takes a tile dictionary and device and returns the indices.
        device (str or torch.device): The device to use for prediction (e.g., 'cuda' or 'cpu').

    Returns:
        pandas.DataFrame: A DataFrame containing information about the predicted indices and area changes for each file.

    Example:
        # Assuming the dataset and pred_func are defined.
        index_predictions_df = predict_indices(dataset, pred_func, 'cuda')
    """
    change_list = []
    for facility in sorted(dataset.get_facilities()):
        facility_file_indices = dataset.get_facility_file_indices(facility)

        for file_index in facility_file_indices:
            file_info = dataset.file_list[file_index]
            predate, postdate = file_info[3:5]
            index_pre, index_post, image_pixels = predict_index(dataset, pred_func, device, file_index, plot=False, output_path=None)
            diff_img = index_post - index_pre
            area_change = diff_img.sum() * 100 # (each pixel is 100 m^2)
            image_area = image_pixels * 100 # each pixel is 100m^2
            prop = float(area_change) / float(image_area)
            change_list.append({ "facility": facility, "index": file_index, "predate": predate, "postdate": postdate, "change_area": area_change, "image_area": image_area, "proportion": prop })
    return pd.DataFrame(change_list)

def predict_masks_validated(dataset, facilities, output_dir=None):
    """
    Predict validated masks for selected facilities using ground truth masks.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing image tiles for prediction.
        facilities (list): A list of facility names for which to predict the masks.
        output_dir (str, optional): The directory to save the generated timeline mask images.
                                    If not provided, the images will not be saved. Defaults to None.

    Returns:
        pandas.DataFrame: A DataFrame containing information about the predicted masks and area changes for each facility.

    Example:
        # Assuming the dataset and facilities list are defined.
        output_directory = '/path/to/output_directory'
        validated_masks_df = predict_masks_validated(dataset, facilities, output_dir=output_directory)
    """

    change_list = []
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    for facility in sorted(facilities):
        facility_file_indices = dataset.get_facility_file_indices(facility)
        cumulative_mask = None
        mask_list = []
        date_list = []
        date_range_list = []
        for file_index in facility_file_indices:
            file_info = dataset.file_list[file_index]
            predate, postdate = file_info[3:5]
            date_range_list.append(f"{predate} - {postdate}")

            mask_id = file_info[5]
            mask_path = os.path.join(dataset.root_dir, 'mask', f'{facility}_{str(mask_id).zfill(4)}.tif')

            if os.path.isfile(mask_path):
                mask, _ = dataset._load_image(mask_path)
                mask = np.clip(mask, a_min=0, a_max=1).astype(np.uint8)

                if cumulative_mask is None:
                    cumulative_mask = mask  # Start with the first mask
                else:
                    cumulative_mask = np.logical_or(cumulative_mask, mask)  # Logical OR operation

                cumulative_area_change = cumulative_mask.sum() * 100  # Total unique area change

                shape = list(mask.shape)
                image_pixels = shape[1] * shape[2]
                area_change = mask.sum() * 100 # (each pixel is 100 m^2)
                image_area = image_pixels * 100 # each pixel is 100m^2
                prop = float(area_change) / float(image_area)
                cumu_prop = float(cumulative_area_change) / float(image_area)
                mask_list.append(mask.squeeze(0))
                date_list.append(postdate)  # We only need postdate for coloring

            # print(f'{facility}:{file_index}:{postdate} - {pixel_change}/{image_area} ({prop}).')
                change_list.append({ "facility": facility, "index": file_index,
                                    "predate": predate, "postdate": postdate,
                                     "change_area": area_change, "image_area": image_area,
                                     "proportion": prop, "cumulative_area": cumulative_area_change,
                                     "cumulative_prop": cumu_prop})
                # break
        # break
        if output_dir is not None:
            create_timeline_mask(date_list, mask_list, date_range_list,
                                path=os.path.join(output_dir, f'{facility}_hybrid_mask_GT.png'))

    df = pd.DataFrame(change_list)
    df['model'] = 'GT'
    return df

def predict_ndtci_masks(dataset, input_dir, output_dir, filter_area=False, facilities=None):
    """
    Predict NDTCI masks for selected facilities using the provided dataset and input directory.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing image tiles for prediction.
        input_dir (str): The directory containing the predicted masks for each facility and file index.
        output_dir (str): The directory to save the average NDTCI GeoTIFFs and masked average NDTCI GeoTIFFs.
        filter_area (bool, optional): Whether to filter the NDTCI masks using area masks if available. Defaults to False.
        facilities (list, optional): A list of facility names for which to predict the NDTCI masks.
                                     If None, all facilities in the dataset will be used.

    Returns:
        pandas.DataFrame: A DataFrame containing information about the predicted NDTCI values for each file.

    Example:
        # Assuming the dataset and input_dir are defined.
        output_directory = '/path/to/output_directory'
        facilities_to_process = ['facility_A', 'facility_B']
        ndtci_predictions_df = predict_ndtci_masks(dataset, input_dir, output_directory, filter_area=True, facilities=facilities_to_process)
    """
    change_list = []
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    if facilities is None:
        facilities = dataset.get_facilities()

    for facility in tqdm(sorted(facilities)):
        facility_file_indices = dataset.get_facility_file_indices(facility)
        if filter_area:
            area_mask_path = os.path.join(dataset.root_dir, 'area_mask', f'{facility}.tif')
            if os.path.isfile(area_mask_path):
                area_mask, _ = dataset._load_image(area_mask_path)
                area_mask = np.clip(area_mask, a_min=0, a_max=1).astype(np.uint8)
                area_mask = area_mask.squeeze(0)

        if 'baseline' not in facility:
            ignore = ['baseline']
        else:
            ignore = []
            
        average_mask = calculate_average_mask(input_dir, facility_name=facility, ignore=ignore)
        x_stable = calculate_x_stable(average_mask, quantile=0.5)
        NDTCI_matrices = []
        if filter_area:
            NDTCI_matrices_masked = []
        meta = None
        date_range_list = []  # This will store the date ranges for each file index
        for file_index in facility_file_indices:
            file_info = dataset.file_list[file_index]
            predate, postdate = file_info[3:5]

            input_path = os.path.join(input_dir, f'pred_{facility}_{file_index}_{predate}_{postdate}.tif')
            
            NDTCI, _, meta = calculate_NDTCI(input_path, x_stable)
            NDTCI_matrices.append(NDTCI)
            shape = list(NDTCI.shape)
            
            if filter_area:
                NDTCI_masked, _, _ = calculate_NDTCI(input_path, x_stable, area_mask=area_mask)
                ndtci_mean = NDTCI_masked.sum() / (shape[0] * shape[1])
                NDTCI_matrices_masked.append(NDTCI_masked)
            else:
                ndtci_mean = NDTCI.sum() / (shape[0] * shape[1])

            change_list.append({"facility": facility, "index": file_index,
                                "predate": predate, "postdate": postdate,
                                "NDTCI": ndtci_mean})

        output_filepath = os.path.join(output_dir, f'{facility}_average_NDTCI.tif')
        write_average_NDTCI(NDTCI_matrices, output_filepath, meta)
        if filter_area:
            output_filepath = os.path.join(output_dir, f'{facility}_masked_average_NDTCI.tif')
            write_average_NDTCI(NDTCI_matrices_masked, output_filepath, meta)
            
    return pd.DataFrame(change_list)


def predict_ndtci_masks_validated(dataset, facilities, output_dir):
    """
    Predict validated NDTCI masks for selected facilities using ground truth masks.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset containing image tiles for prediction.
        facilities (list): A list of facility names for which to predict the NDTCI masks.
        output_dir (str): The directory to save the generated mean-NDTCI GeoTIFFs.

    Returns:
        pandas.DataFrame: A DataFrame containing information about the predicted NDTCI values for each file.

    Example:
        # Assuming the dataset and facilities list are defined.
        output_directory = '/path/to/output_directory'
        facilities_to_process = ['facility_A', 'facility_B']
        ndtci_validated_df = predict_ndtci_masks_validated(dataset, facilities_to_process, output_directory)
    """

    change_list = []

    for facility in sorted(facilities):
        facility_file_indices = dataset.get_facility_file_indices(facility)
        if 'alt' not in facility:
            ignore = ['alt']
        else:
            ignore = []
        average_mask = calculate_average_mask(os.path.join(dataset.root_dir, 'mask'), facility_name=facility, ignore=ignore, prefix='')
        x_stable = calculate_x_stable(average_mask, quantile=0.5)
        NDTCI_matrices = []
        for file_index in facility_file_indices:
            file_info = dataset.file_list[file_index]
            predate, postdate = file_info[3:5]

            mask_id = file_info[5]
            mask_path = os.path.join(dataset.root_dir, 'mask', f'{facility}_{str(mask_id).zfill(4)}.tif')
            if os.path.isfile(mask_path):
                NDTCI, _, meta = calculate_NDTCI(mask_path, x_stable)
                NDTCI_matrices.append(NDTCI)
                shape = list(NDTCI.shape)
                ndtci_mean = NDTCI.sum() / (shape[0] * shape[1])
                change_list.append({"facility": facility, "index": file_index,
                                    "predate": predate, "postdate": postdate,
                                     "NDTCI": ndtci_mean})        

        output_filepath = os.path.join(output_dir, f'{facility}_GT_average_NDTCI.tif')
        write_average_NDTCI(NDTCI_matrices, output_filepath, meta)
        
    df = pd.DataFrame(change_list)
    df['model'] = 'GT'
    return df

def create_correlation_plot(df_path, var, title, output_path, gt=False):
    """
    Create a correlation plot based on the data from the mask change prediction dataframe.

    Parameters:
        df_path (str): The file path to the CSV file containing the DataFrame.
        var (str): The variable/column name in the DataFrame to be used for correlation analysis.
        title (str): The title of the correlation plot.
        output_path (str): The file path to save the generated correlation plot.
        gt (bool, optional): Whether to include the 'GT' (ground truth) model in the correlation analysis. Defaults to False.

    Example:
        # Assuming the DataFrame file and other parameters are defined.
        correlation_var = 'proportion'
        correlation_title = 'Correlation Plot for Proportion'
        correlation_output = '/path/to/correlation_plot.png'
        include_gt_model = True
        create_correlation_plot(df_file, correlation_var, correlation_title, correlation_output, gt=include_gt_model)
    """
    df = pd.read_csv(df_path)
    # Pivot the DataFrame so that each model's 'proportion' forms a column
    df_pivot = df.pivot_table(index=['facility', 'predate', 'postdate'], 
                              columns='model', values=var).reset_index()
    models = ['DDPMCD', 'TinyCD', 'LSNet']
    if gt:
        models.append('GT')
    df_pivot['Median'] = df_pivot[models].loc[:, df_pivot.dtypes != 'object'].median(axis=1)

    # Now, compute the correlation matrix specifying 'numeric_only=True'
    correlation_matrix = df_pivot.corr(numeric_only=True)

    # Also, compute a matrix of p-values
    p_value_matrix = correlation_matrix.copy()
    for col in correlation_matrix.columns:
        for row in correlation_matrix.index:
            if col != row:  # Skip when column and row are the same
                temp_df = df_pivot[[col, row]].dropna()
                try:
                    r, p_value = stats.pearsonr(temp_df[col].values, temp_df[row].values)
                    p_value_matrix.loc[row, col] = p_value
                except ValueError as e:
                    print(f"Failed to calculate pearson correlation for columns '{col}' and '{row}': {str(e)}")
                    p_value_matrix.loc[row, col] = np.nan
            else:  # If column and row are the same, set p-value to NaN (as it's not meaningful in this context)
                p_value_matrix.loc[row, col] = np.nan

    # Replace p-values with asterisks to denote their significance levels
    significance_level = 0.05  # Set a significance level
    p_value_matrix = p_value_matrix.applymap(lambda x: '***' if x < significance_level/100 else '**' if x < significance_level/10 else '*' if x < significance_level else '')

    # Combine correlation values with significance annotations
    combined_matrix = correlation_matrix.applymap('{:.2f}'.format) + "\n" + p_value_matrix
    # Create a heatmap of correlation coefficients, annotated with significance levels
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=combined_matrix, fmt='s', square=True, cmap='coolwarm', 
                xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns,
                vmin=0.4, vmax=1)

    plt.title(title)
    
    # Add a description of significance levels
    plt.text(0.33, 0.1, f'Significance levels: *: p<{significance_level:.2f}, **: p<{significance_level/10:.3f}, ***: p<{significance_level/100:.4f}', 
             size=10, ha="center", transform=plt.gcf().transFigure)
    plt.savefig(output_path)
    plt.show()

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def calculate_rmse(df_path, var):
    """
    Calculate the root mean squared error (RMSE) for each model's predictions compared to ground truth.

    Parameters:
        df_path (str): The file path to the CSV file containing the DataFrame with prediction data.
        var (str): The variable/column name in the DataFrame to be used for RMSE calculation.

    Returns:
        pandas.DataFrame: A DataFrame containing the RMSE values for each model.

    Example:
        # Assuming the DataFrame file and the variable name are defined.
        rmse_var = 'proportion'
        df_file_path = '/path/to/df_file.csv'
        rmse_df = calculate_rmse(df_file_path, rmse_var)
    """
    df = pd.read_csv(df_path)
    # Pivot the DataFrame so that each model's 'proportion' forms a column
    df_pivot = df.pivot_table(index=['facility', 'predate', 'postdate'], 
                              columns='model', values=var).reset_index()
    models = ['DDPMCD', 'TinyCD', 'LSNet', 'GT']
    df_pivot['Median'] = df_pivot[models].loc[:, df_pivot.dtypes != 'object'].median(axis=1)
    df_pivot = df_pivot.dropna()
    models = ['DDPMCD', 'TinyCD', 'LSNet', 'Median']
    # Initialize an empty DataFrame to store the metrics
    metrics = []
    for model in models:
        # Calculate the RMSE
        rmse_value = rmse(df_pivot['GT'], df_pivot[model])

        # Append the metrics to the list
        metrics.append({'model': model, 'RMSE': rmse_value})
    
    return pd.DataFrame(metrics)

def plot_rmse(df_path1, df_path2, var, title, output_path):
    """
    Plot the root mean squared error (RMSE) comparison between two sets of predictions.

    Parameters:
        df_path1 (str): The file path to the first CSV file containing the DataFrame for the first set of predictions.
        df_path2 (str): The file path to the second CSV file containing the DataFrame for the second set of predictions.
        var (str): The variable/column name in the DataFrame to be used for RMSE calculation.
        title (str): The title of the RMSE comparison plot.
        output_path (str): The file path to save the generated RMSE comparison plot.
        
    Example:
        # Assuming the DataFrame files and other parameters are defined.
        rmse_var = 'proportion'
        df_file_path1 = '/path/to/df_file1.csv'
        df_file_path2 = '/path/to/df_file2.csv'
        rmse_plot_title = 'RMSE Comparison between Two Sets of Predictions'
        rmse_output_path = '/path/to/rmse_comparison_plot.png'
        plot_rmse(df_file_path1, df_file_path2, rmse_var, rmse_plot_title, rmse_output_path)
    """
    df_metrics1 = calculate_rmse(df_path1, var)
    df_metrics2 = calculate_rmse(df_path2, var)

    models = ['DDPMCD', 'TinyCD', 'LSNet', 'Median']
    pos = list(range(len(df_metrics1['RMSE'])))
    width = 0.2
    print(df_metrics1)
    print(df_metrics2)
    fig, ax = plt.subplots(figsize=(7,5))

    plt.bar(pos,
            df_metrics1['RMSE'],
            width,
            # alpha=0.5,
            # color='#EE3224',
            label='with mask')

    plt.bar([p + width for p in pos],
            df_metrics2['RMSE'],
            width,
            label='no mask')

    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.set_xticks([p + 0.5 * width for p in pos])
    ax.set_xticklabels(models)
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(output_path)
    plt.show()