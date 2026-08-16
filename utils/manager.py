from abc import ABC, abstractmethod
import os 
import torch

from utils.results import RunResult
from utils.utils import ModelFileNotFoundError
    
class STModelManager(ABC):
    """
    Abstract spatial-temporal model from which all models must be derived
    """

    def __init__(self, model=None):
        """
        Constructor for STModelManager

        Parameters
        ----------
        model : torch.nn.Module, optional
            The torch model to manage, by default None
        """
        self.model = model 

    def set_model(self, model):
        """
        Setter for model

        Parameters
        ----------
        model : torch.nn.Module
            The torch model to manage
        """
        self.model = model

    def has_model(self):
        """
        Returns whether a model has been specified

        Returns
        -------
        bool
            Whether a model has been specified
        """
        return self.model is not None

    def save_model(self, model_dir, epoch=None):
        """
        Saves the model in a specified directory

        Parameters
        ----------
        model_dir : str
            The directory in which to save the model
        epoch : int, optional
            The epoch associated with the model, by default None
        """
        if model_dir is None:
            return
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        epoch = str(epoch) if epoch else type(self.model).__name__
        file_name = os.path.join(model_dir, epoch + '.pt')
        with open(file_name, 'wb') as f:
            torch.save(self.model, f)

    def load_model(self, model_dir, epoch=None):
        """
        Loads a model from a .pt file

        Parameters
        ----------
        model_dir : str
            The directory in which the model file is located
        epoch : int, optional
            The epoch from which to load the model, by default None

        Raises
        ------
        ModelFileNotFoundError
            When the model directory cannot be found
        ModelFileNotFoundError
            When the model file cannot be found
        """
        if not model_dir:
            return
        epoch = str(epoch) if epoch else type(self.model).__name__
        file_name = os.path.join(model_dir, epoch + '.pt')
        if not os.path.exists(model_dir):
            raise ModelFileNotFoundError(f"Could not locate directory '{model_dir}'")
        if not os.path.exists(file_name):
            raise ModelFileNotFoundError(f"Could not locate model file '{file_name}'")
        with open(file_name, 'rb') as f:
            self.model = torch.load(f)

    def run_pipeline(self, config):
        """
        Executes the machine learning pipeline for the given model

        Parameters
        ----------
        ExperimentConfig
            An ExperimentConfig object containing the parameters for the model and pipeline

        Returns
        -------
        RunResult
            A RunResult containing the results collected in the pipeline
        """
        if not self.has_model():
            self.set_model(config.model_type(**config.get_model_params()))
        train, valid, test = self.preprocess(**config.get_preprocessing_params())
        train_results = self.train_model(train, valid, **config.get_training_params())
        test_results = self.test_model(test, **config.get_testing_params())
        result = RunResult(
            {**train_results, **test_results}    
        )
        return result

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        Abstract method for executing the preprocessing of the data

        Returns
        -------
        tuple(any)
            A tuple of objects with the respective processed training, validation, and testing data 
        """
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        """
        Abstract method for executing the model training and validation

        Returns
        -------
        dict
            A dictionary of pandas.DataFrame objects with the training results
        """
        pass

    @abstractmethod
    def test_model(self, *args, **kwargs):
        """
        Abstract method for executing the model testing

        Returns
        -------
        dict
            A dictionary of pandas.DataFrame objects with the testing results
        """
        pass

