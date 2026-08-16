import os
import json
import pickle

# TODO Add experiment config specific functionality if necessary (e.g. resolve classes)
def load_experiment_config(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not locate experiment config file '{filepath}'")
    with open(filepath, "r") as json_file:
        return json.load(json_file)

class PickleMixin():
    """
    A mixin for providing general serialization and deserialization of objects
    """

    def save_to(self, directory, filename):
        """
        Pickles current instance to file in specified directory

        Parameters
        ----------
        directory : str
            Directory to save the file in
        filename : str
            The name of the file
        """
        path = filename
        if directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
            path = os.path.join(directory, path)
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_from(cls, directory, filename):
        """
        Unpickles a pickle file and returns the object

        Parameters
        ----------
        directory : str
            Directory in which to find the file
        filename : str
            The name of the file

        Returns
        -------
        any
            The object instance loaded from the pickle file

        Raises
        ------
        FileNotFoundError
            When the directory specified could not be found
        FileNotFoundError
            When the file specified could not be found
        TypeError
            When the loaded object does not match the calling class
        """
        path = filename
        if directory is not None:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory '{directory}' could not be found")
            path = os.path.join(directory, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' could not be found")
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        if not isinstance(obj, cls):
            raise TypeError(f"The object loaded from '{filename}' is not of type '{cls}'")
        return obj

            