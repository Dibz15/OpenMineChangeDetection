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
            predate, postdate = file_info[3:5]
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
    df_metrics1 = calculate_rmse(df_path1, var)
    df_metrics2 = calculate_rmse(df_path2, var)

    models = ['DDPMCD', 'TinyCD', 'LSNet', 'Median']

    # Setting the positions and width for the bars
    pos = list(range(len(df_metrics1['RMSE'])))
    width = 0.2
    print(df_metrics1)
    print(df_metrics2)
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(7,5))

    # Create a bar with RMSE data from the first DataFrame
    plt.bar(pos,
            df_metrics1['RMSE'],
            width,
            # alpha=0.5,
            # color='#EE3224',
            label='with mask')

    # Create a bar with RMSE data from the second DataFrame,
    # in position pos + some width buffer
    plt.bar([p + width for p in pos],
            df_metrics2['RMSE'],
            width,
            # alpha=0.5,
            # color='#F78F1E',
            label='no mask')

    # Set the y axis label
    ax.set_ylabel('RMSE')

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 0.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(models)

    # Adding the legend and showing the plot
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(output_path)
    plt.show()

# Use the function as:
# plot_rmse(df_path1, df_path2, var, title)