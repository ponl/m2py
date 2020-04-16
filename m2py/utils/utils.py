import numpy as np


def generate_chips_from_data(data, chip_size, stride):
    """
    Finds outliers from data

    Parameters
    ----------
        data : NumPy Array
            SPM data supplied by the user
        chip_size : int
            size of generated chips
        stride: int
            number of pixels skipped over to generate adjacent chips

    Returns
    ----------
        chips : dict
            dictionary of partition indices mapping to actual chips
    """
    len_shape = len(data.shape)
    if len_shape == 2:
        c = 1
        h, w = data.shape
        data = data.reshape(h, w, c)
    elif len_shape == 3:
        h, w, c = data.shape
    else:
        print("Input data array has more than 3 valid axes.")
        return None

    h_limit = h - chip_size
    w_limit = w - chip_size

    chips = {}
    for i in range(0, h_limit, stride):
        for j in range(0, w_limit, stride):
            chip = data[i : i + chip_size, j : j + chip_size, :]

            key = (i, j)
            chips[key] = reshape_image(chip, chip_size, chip_size, c, len_shape)

    # Generate last column of chips
    for i in range(0, h_limit, stride):
        chip = data[i : i + chip_size, w_limit : w_limit + chip_size, :]

        key = (i, w_limit)
        chips[key] = reshape_image(chip, chip_size, chip_size, c, len_shape)

    # Generate last row of chips
    for j in range(0, w_limit, stride):
        chip = data[h_limit : h_limit + chip_size, j : j + chip_size, :]

        key = (h_limit, j)
        chips[key] = reshape_image(chip, chip_size, chip_size, c, len_shape)

    # Generate last chip
    chip = data[h_limit : h_limit + chip_size, w_limit : w_limit + chip_size, :]

    key = (h_limit, w_limit)
    chips[key] = reshape_image(chip, chip_size, chip_size, c, len_shape)

    print(f"There were {len(chips)} chips generated.")

    return chips


def reshape_image(image, image_height, image_width, num_channels, len_shape):
    """
    Reshape image into input dimensions

    Parameters
    ----------
        image : NumPy array
            image to reshape
        image_height : int
            image height
        image_width : int
            image width
        num_channels : int
            number of channels in image
        len_shape : int
            number of axes in input image

    Returns
    ----------
        image : NumPy array
            reshaped image
    """
    if len_shape == 3:
        image = image.reshape(image_height, image_width, num_channels)
    else:
        image = image.reshape(image_height, image_width)

    return image


def get_stride_from_chips(chips):
    """
    Stitch up chips into a full array

    Parameters
    ----------
        chips : dict
            dictionary of partition indices mapping to actual chips

    Returns
    ----------
        stride: int
            number of pixels skipped over to generate adjacent chips
    """
    keys = list(chips.keys())
    sorted_keys = sorted(keys)

    if len(sorted_keys) == 1:
        return 0

    first_key = sorted_keys[0]
    second_key = sorted_keys[1]

    stride = second_key[1] - first_key[1]
    return stride


def stitch_up_chips(chips):
    """
    Stitch up chips into a full array

    Parameters
    ----------
        chips : dict
            dictionary of partition indices mapping to actual chips

    Returns
    ----------
        full_image: NumPy Array
            stitched up image made up of chips
    """
    stride = get_stride_from_chips(chips)

    first_chip = chips[(0, 0)]
    chip_size = first_chip.shape[0]
    len_shape = len(first_chip.shape)
    num_channels = first_chip.shape[2] if len_shape == 3 else 1

    full_image_height = chip_size + max(chips.keys(), key=lambda k: k[0])[0]
    full_image_width = chip_size + max(chips.keys(), key=lambda k: k[1])[1]

    full_image = np.zeros((full_image_height, full_image_width, num_channels), dtype="int64")
    intersection = np.zeros((full_image_height, full_image_width, 1), dtype="int64")
    for (i, j), chip in chips.items():
        full_image[i : i + chip_size, j : j + chip_size, :] += chip.reshape(chip_size, chip_size, num_channels)
        intersection[i : i + chip_size, j : j + chip_size, 0] += 1

    full_image = full_image / intersection
    full_image = full_image.astype("int64")
    full_image = reshape_image(full_image, full_image_height, full_image_width, num_channels, len_shape)

    return full_image
