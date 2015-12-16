from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

class OHStratifiedShuffleSplit(StratifiedShuffleSplit):
    def __init__(self, fm_decoder, n_iter=5, test_size=0.2, train_size=None,
                 random_state=None):
        self.fm_decoder = fm_decoder
        StratifiedShuffleSplit.__init__(
            self,
            n_iter=n_iter,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)

    def _iter_indices(self, X, y, labels=None):
        samples, features = self.fm_decoder.fm_to_indices(X)
        for train, test in super(OHStratifiedShuffleSplit, self)._iter_indices(
                X, samples):
            yield train, test


class OHStratifiedKFold(StratifiedKFold):
    def __init__(self, fm_decoder, n_iter=5, n_folds=3,
                 random_state=None):
        self.fm_decoder = fm_decoder
        StratifiedKFold.__init__(
            self,
            n_folds=n_folds,
            random_state=random_state)

    def _iter_test_masks(self, X, y, labels=None):
        samples, features = self.fm_decoder.fm_to_indices(X)
        for mask in StratifiedKFold._iter_test_masks(self,
                                                     X, samples):
            yield mask