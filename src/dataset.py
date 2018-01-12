import numpy as np


class SmallNORBDataset:
    """SmallNORBDataset

    Args:
        train_npz (str): Path to the npz file that contains keys
            - lt: Training images from the first camera with shape
                (24300, 96, 96).
            - rt: Training images from the second camera with shape
                (24300, 96, 96).
            - category: Category of the images with shape (24300,).
        test_npz (str): Path to the npz file that contains keys
            - lt: Testing images from the first camera with shape
                (24300, 96, 96).
            - rt: Testing images from the second camera with shape
                (24300, 96, 96).
            - category: Category of the images with shape (24300,).
    """

    def __init__(self, train_npz, test_npz):
        self.train_data_raw = np.load(train_npz)
        self.test_data_raw = np.load(test_npz)

    def _preprocess_image(self, imgs):
        # convert color [0, 255] to [-1, +1]
        imgs = imgs.astype(np.float32)
        imgs = (imgs - 128) / 128

        # downsample to 48x48
        imgs = imgs[:, ::2, ::2]
        assert imgs.shape[1:] == (48, 48)

        # expand channel dimension
        imgs = np.expand_dims(imgs, -1)
        return imgs

    def get_train_valid(self, valid_instance=4):
        """Get train and valid data.

        Args:
            valid_category (int): Category to be used as validation dataset.
        """
        train_indices = np.where(
            self.train_data_raw['instance'] != valid_instance)
        train = {
            'x': np.concatenate([self.train_data_raw['lt'][train_indices],
                                 self.train_data_raw['rt'][train_indices]],
                                axis=0),
            'y': np.concatenate(
                [self.train_data_raw['category'][train_indices],
                 self.train_data_raw['category'][train_indices]], axis=0)
        }

        valid_indices = np.where(
            self.train_data_raw['instance'] == valid_instance)
        valid = {
            'x': np.concatenate([self.train_data_raw['lt'][valid_indices],
                                 self.train_data_raw['rt'][valid_indices]],
                                axis=0),
            'y': np.concatenate(
                [self.train_data_raw['category'][valid_indices],
                 self.train_data_raw['category'][valid_indices]], axis=0)
        }

        train['x'] = self._preprocess_image(train['x'])
        valid['x'] = self._preprocess_image(valid['x'])

        return train, valid
