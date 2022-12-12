import tensorflow as tf

class StockModel:

    def __init__(self, filePath, model: tf.keras.Model = None):
        self.filePath = filePath
        if model is None:
            self.load_model()
        else:
            self.model = model

    def save_model(self):
        self.model.save(filepath=self.filePath)

    def load_model(self):
        self.model = tf.keras.models.load_model(filepath=self.filePath)

    def train_model(self):

