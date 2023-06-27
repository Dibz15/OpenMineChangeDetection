import os
import shutil
import zipfile

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def copy_file(source_path, destination_path):
    shutil.copy(source_path, destination_path)

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def move_folder(source_path, destination_path):
    shutil.move(source_path, destination_path)

def copy_and_unzip(local_path, drive_path, archive_name):
    print(f"Copying the archive file '{archive_name}' from Google Drive to local folder...")
    copy_file(os.path.join(drive_path, archive_name), os.path.join(local_path, archive_name))
    print("Archive file copied successfully.")

    print(f"Unzipping the archive file '{archive_name}'...")
    unzip_file(os.path.join(local_path, archive_name), local_path)
    print("Archive file unzipped successfully.")

def load_and_prepare_oscd(local_path="/content/datasets", drive_path="/content/drive/MyDrive/2023_dissertation/dataset_archives/"):
    if os.path.exists(os.path.join(local_path, "OSCD")):
        print("An 'OSCD' folder already exists in the local path. Skipping dataset loading and preparation.")
        return
    
    create_folder_if_not_exists(local_path)

    OSCD = "OSCD_Daudt_2018_full.zip"
    copy_and_unzip(local_path, drive_path, OSCD)

    print("Moving the extracted folders and renaming if necessary...")
    move_folder(os.path.join(local_path, "Onera"), os.path.join(local_path, "OSCD"))
    print("Main folder renamed successfully.")

    # Define the folder mappings
    folder_mappings = {
        "images": "Onera Satellite Change Detection dataset - Images",
        "train_labels": "Onera Satellite Change Detection dataset - Train Labels",
        "test_labels": "Onera Satellite Change Detection dataset - Test Labels"
    }

    # Move and rename the extracted folders
    oscd_path = os.path.join(local_path, "OSCD")
    for source_folder, destination_folder in folder_mappings.items():
        source_path = os.path.join(oscd_path, source_folder)
        destination_path = os.path.join(oscd_path, destination_folder)
        print(f"Renaming folder '{source_folder}' to '{destination_folder}'...")
        move_folder(source_path, destination_path)
        print(f"Folder '{source_folder}' renamed successfully to '{destination_folder}'.")
    
    print("Dataset loading and preparation complete.")

def load_and_prepare_omcd(local_path="/content/datasets", drive_path="/content/drive/MyDrive/2023_dissertation/dataset_archives/"):
    if os.path.exists(os.path.join(local_path, "OMCD")):
        print("An 'OMCD' folder already exists in the local path. Skipping dataset loading and preparation.")
        return

    create_folder_if_not_exists(local_path)

    OMCD = "OMCD_Li_2023.zip"
    copy_and_unzip(local_path, drive_path, OMCD)

    print("Moving the extracted folders and renaming to OMCD...")
    move_folder(os.path.join(local_path, "open-pit mine change detection dataset"), os.path.join(local_path, "OMCD"))
    print("Main folder renamed successfully.")

    # # Define the folder mappings
    # folder_mappings = {
    #     "images": "Onera Satellite Change Detection dataset - Images",
    #     "train_labels": "Onera Satellite Change Detection dataset - Train Labels",
    #     "test_labels": "Onera Satellite Change Detection dataset - Test Labels"
    # }

    # # Move and rename the extracted folders
    # oscd_path = os.path.join(local_path, "OSCD")
    # for source_folder, destination_folder in folder_mappings.items():
    #     source_path = os.path.join(oscd_path, source_folder)
    #     destination_path = os.path.join(oscd_path, destination_folder)
    #     print(f"Renaming folder '{source_folder}' to '{destination_folder}'...")
    #     move_folder(source_path, destination_path)
    #     print(f"Folder '{source_folder}' renamed successfully to '{destination_folder}'.")

    print("Dataset loading and preparation complete.")
