class StandardScaler:
    """ Used to calculate mean, std and normalize data. """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data, traj=False):
        """ Calculate mean and std for given data."""
        if not traj:
            self.mean = data.mean(0, keepdim=True)  # calculate mean among batch
            self.std = data.std(0, keepdim=True)
            self.std[self.std < 1e-12] = 1.0
        else:
            self.mean = data.mean([0, 1], keepdim=True)  # calculate mean among batch
            self.std = data.std([0, 1], keepdim=True)
            self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """ Normalization. """
        return (data - self.mean) / self.std

    def transform_std_only(self, data):
        return data / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean
