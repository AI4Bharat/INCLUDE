from dataclasses import asdict
import pickle

import xgboost


class Xgboost:
    def __init__(self, config):
        config = asdict(config)
        self.model = xgboost.XGBClassifier(use_label_encoder=False, **config)

    def __call__(self, x):
        return self.model.predict(x)

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            verbose=True,
        )

    def save(self, save_path):
        pickle.dump(self.model, open(save_path, "wb"))

    def load(self, load_path):
        self.model = pickle.load(open(load_path, "rb"))
