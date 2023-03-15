import keras


class StockModel:

    def __init__(self, name, model: keras.Sequential = None):
        self.name = name
        if model is None:
            self.load_model()
        else:
            self.model = model

    def save_model(self):
        self.model.save(filepath=f"Models/{self.name}")

    def load_model(self):
        self.model = keras.models.load_model(filepath=f"Models/{self.name}")

    def update_model(self, model: keras.Sequential = None):
        self.model = model
