import os
import dicomsdl
import pydicom
# from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cv2

# Define directory paths
directory = 'your_target_folder' 
output_directory = 'datapre/pretrain_data/' 

def np_CountUpContinuingOnes(b_arr):
    # indice continuing zeros from left side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    # from right side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def ExtractBreast(img):
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)  # To detect backgrounds easily
    height, _ = img.shape

    # whether each col is non-constant or not
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    # whether each row is non-constant or not
    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = np_CountUpContinuingOnes(b_arr)
    # longest should be the breast
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    return img_copy[row_ind][:, col_ind]

def save_imgs(in_path, out_path):
    try:
        dicom = dicomsdl.open(in_path)
        data = dicom.pixelData()
        data = data[5:-5, 5:-5]

        if dicom.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        img = ExtractBreast(data)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Save image
        cv2.imwrite(out_path, img)
        print(f"Successfully saved image: {out_path}")

    except Exception as e:
        print(f"Error processing {in_path}: {e}")



# Create an empty list to store file information
file_list = []


for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a DICOM file
        if file.lower().endswith('.dcm'):
            # Read the DICOM file
            dicom_path = os.path.join(root, file)
            try:
                # Force read DICOM file
                dicom_data = pydicom.dcmread(dicom_path, force=True)

                # Extract metadata
                # view_position = dicom_data.ViewPosition if 'ViewPosition' in dicom_data else 'N/A'
                # image_laterality = dicom_data.ImageLaterality if 'ImageLaterality' in dicom_data else 'N/A'

                # Save PNG file
                # Get relative path
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                # Create PNG file path (keep original structure, only change extension)
                png_path = os.path.join(output_directory, os.path.splitext(relative_path)[0] + '.png')

                # Ensure output directory exists
                png_output_dir = os.path.dirname(png_path)
                if not os.path.exists(png_output_dir):
                    os.makedirs(png_output_dir)

                save_imgs(dicom_path, png_path)
                # print('save png:', png_path)

                # Add file info to the list
                file_list.append({
                    'file_name': file,
                    'original_path': relative_path,
                    'image_path': png_path,
                    # 'view': view_position,
                    # 'laterality': image_laterality
                })

            except pydicom.errors.InvalidDicomError as e:
                print(f"Unable to read DICOM file {dicom_path}: {e}")
            except Exception as e:
                print(f"Error processing file {dicom_path}: {e}")
# Create DataFrame
df = pd.DataFrame(file_list)

# Save to CSV file
df.to_csv('predata.csv', index=False)

