class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
