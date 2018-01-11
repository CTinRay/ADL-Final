import numpy as np


def BatchGenerator(X, y, batch_size, shuffle=True):
    if shuffle:
        indices = np.random.permutation(X.shape[0])
        X = X[indices]

        if y is not None:
            y = y[indices]

    for i in range(X.shape[0] // batch_size + 1):
        batch = {}
        batch['x'] = X[i * batch_size: (i + 1) * batch_size]

        lengths = np.sum(~np.isnan(batch['x'][:, :, 0]), axis=-1)
        max_length = np.max(lengths)
        # batch['x'] = batch['x'][:, :max_length, :]
        if y is not None:
            batch['y'] = y[i * batch_size: (i + 1) * batch_size]
            # batch['y'] = batch['y'][:, :max_length]

        yield batch
