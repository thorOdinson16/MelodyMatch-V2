class EarlyStopping:
    """
    Early stopping based on a validation metric.

    Higher values are assumed to be better.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:

        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop

    def reset(self):
        self.best_score = None
        self.counter = 0
        self.should_stop = False