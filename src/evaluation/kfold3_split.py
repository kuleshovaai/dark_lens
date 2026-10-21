import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
)


class KFold3:
    """
    Split dataset into train, val and test subsets.

    Ensures that unique `context` values never leak into the test set.
    No group constraints are applied between train and validation.

    With ``n_splits=1``, performs a single stratified (or group) holdout
    using ``test_size``, then carves validation from the remainder with
    ``val_size``.
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_size: float = 0.0,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        self.n_splits = n_splits
        self.val_size = val_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def _split_val(self, train_val_idx, y):
        if self.val_size == 0:
            return train_val_idx, np.array([], dtype=train_val_idx.dtype)
        if self.val_size == 1:
            return np.array([], dtype=train_val_idx.dtype), train_val_idx
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y[train_val_idx],
        )
        return train_idx, val_idx

    def _single_split(self, X, y, groups=None):
        indices = np.arange(len(y))
        if groups is None or len(np.unique(groups)) == len(groups):
            # Unique groups (or no groups): stratified holdout is fine.
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=y,
            )
        else:
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_val_idx, test_idx = next(gss.split(X, y, groups))

        train_idx, val_idx = self._split_val(train_val_idx, y)
        yield train_idx, val_idx, test_idx

    def split(self, X, y, groups=None):
        np.random.seed(self.random_state)

        if self.n_splits == 1:
            yield from self._single_split(X, y, groups)
            return

        if groups is None:
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            splits = cv.split(X, y)
        else:
            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            splits = cv.split(X, y, groups)

        for train_val_idx, test_idx in splits:
            train_idx, val_idx = self._split_val(train_val_idx, y)
            yield train_idx, val_idx, test_idx
