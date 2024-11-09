import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(file_name)
        self.pipeline = None

    def create_pipeline(self, steps):
        self.pipeline = Pipeline(steps)

    def fit_transform(self):
        if self.pipeline is not None:
            return self.pipeline.fit_transform(self.data)
        else:
            raise ValueError("Pipeline has not been created. Use create_pipeline method to create one.")

    def add_preprocessing_step(self, name, transformer, columns):
        if self.pipeline is None:
            self.pipeline = ColumnTransformer([(name, transformer, columns)])
        else:
            self.pipeline.transformers.append((name, transformer, columns))

# Example usage:
# preprocessor = DataPreprocessor('data.csv')
# preprocessor.add_preprocessing_step('scale', StandardScaler(), ['feature1', 'feature2'])
# preprocessor.add_preprocessing_step('onehot', OneHotEncoder(), ['category'])
# preprocessor.create_pipeline([('preprocess', preprocessor.pipeline)])
# processed_data = preprocessor.fit_transform()