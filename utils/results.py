import pandas as pd
from utils.pickl_mixin import PickleMixin

class Result(PickleMixin):
    """
    Class for storing and managing results (a dictionary of related DataFrames)
    """

    def __init__(self, results={}):
        """
        Constructor for Result

        Parameters
        ----------
        results : dict, optional
            Results to store, by default {}
        """
        self._results = {}
        for key, frame in results.items():
            self.add_dataframe(frame, key)

    def add_dataframe(self, dataframe, key):
        """
        Adds copy of the specified DataFrame

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The DataFrame to add
        key : any
            The key to store the DataFrame under
        """
        self._results[key] = dataframe.copy(deep=True)

    def get_dataframe(self, key):
        """
        Returns a copy of the specified DataFrame

        Parameters
        ----------
        key : any
            The key under which the DataFrame is stored

        Returns
        -------
        pandas.DataFrame
            A copy of the corresponding DataFrame
        """
        return self._results.get(key).copy(deep=True)

    def get_dataframes(self):
        """
        Returns deep copy of the results dictionary

        Returns
        -------
        dict
            A deep of the results dictionary
        """
        return {key: dataframe.copy(deep=True) for key, dataframe in self._results.items()}

    def get_dataframe_names(self):
        """
        Returns a list of the keys/names of the stored DataFrames

        Returns
        -------
        list
            A list of the keys
        """
        return self._results.keys()

    def copy(self):
        """
        Returns a deep copy of the Result instance

        Returns
        -------
        Result
            A deep copy of the Result
        """
        return Result(self._results)

    def __str__ (self):
        """
        To string dunder method

        Returns
        -------
        str
            A user-friendly string representing the Result
        """
        return str(self._results)

    def __repr__(self):
        """
        String representation dunder method

        Returns
        -------
        str
            A string representing the Result
        """
        return "Result(\n" + str(self) + "\n)"

class RunResult(Result):
    """
    Class representing the results of an experiment run
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor for RunResult
        """
        super().__init__(*args, **kwargs)

    def copy(self):
        """
        Returns deep copy of RunResult instance

        Returns
        -------
        RunResult
            A deep copy of the RunResult
        """
        return RunResult(self._results)

class ExperimentResult(Result):
    """
    Class representing the results of an experiment (combined/aggregated runs)
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor for ExperimentResult
        """
        super().__init__(*args, **kwargs)

    # TODO Generalize to apply any aggregation function(s) across each dataframe
    def aggregate(self, group_by, columns=None, which=None, join=True):
        """
        Returns Result which represents aggregated run data

        Parameters
        ----------
        group_by : str
            The column by which to form groups
        columns : list[str]
            The columns to aggregate and include, optional, by default all
        which : list[str], optional
            The keys of the dataframes to perform aggregation on, by default all
        join : bool, optional
            Whether to merge frames if multiple aggregation methods are applied, by default True

        Returns
        -------
        ExperimentResult
            The aggregated ExperimentResults
        """
        aggregated = ExperimentResult()
        for key, frame in self.get_dataframes().items():
            # Drop if which specified and key not in which 
            if which is not None and key not in which:
                continue
            # Otherwise, group frame, select columns if specified and calculate mean/std-dev
            grouped = frame.groupby(by=group_by)
            if columns is not None:
                grouped = grouped[columns]
            means = grouped.mean()
            devs = grouped.std()
            if join:
                means.columns = [(name + "_mean") for name in means.columns if name != group_by]
                devs.columns = [(name + "_std_dev") for name in devs.columns if name != group_by]
                aggregated.add_dataframe(means.join(devs, on=group_by).reset_index(), key)
            else :
                aggregated.add_dataframe(means, "mean_" + key)
                aggregated.add_dataframe(devs, "std_dev_" + key)
        return aggregated

    def copy(self):
        """
        Returns deep copy of ExperimentResult instance

        Returns
        -------
        ExperimentResult
            A deep copy of the ExperimentResult
        """
        return ExperimentResult(self._results)

class ResultSet(PickleMixin):
    """
    Class for storing and managing many Result objects
    """

    def __init__(self):
        """
        Constructor for ResultSet
        """
        self._results = {}
        self._count = 0

    def add_result(self, result, key=None):
        """
        Adds copy of result to set

        Parameters
        ----------
        result : Result
            The result to add to the set
        key : any, optional
            The key under which to store the result, 
            by default an id is assigned
        """
        self._count += 1
        self._results[key if key is not None else str(self._count)] = result.copy()

    def add_results(self, results):
        """
        Adds results from dictionary to set

        Parameters
        ----------
        results : Union[dict, list]
            A dictionary/list containing the Result objects

        Raises
        ------
        TypeError
            If results is not a dict or list
        """
        if isinstance(results, dict):
            for key, result in results.items():
                self.add_result(result, key)
        elif isinstance(results, list):
            for result in results:
                self.add_result(result)
        else:
            raise TypeError(f"Expecting list or dict, got {type(results)}")

    def get_result(self, key):
        """
        Returns copy of Result stored under specified key

        Parameters
        ----------
        key : any
            The key under which the Result is stored

        Returns
        -------
        Result
            The Result object
        """
        return self._results.get(key).copy()

    def get_results(self):
        """
        Returns a deep copy of the Results dictionary

        Returns
        -------
        dict
            A shallow deep of the Results dictionary
        """
        return {key: result.copy() for key, result in self._results.items()}

    def copy(self):
        """
        Returns deep copy of ResultSet

        Returns
        -------
        ResultSet
            A deep copy of the current ResultSet
        """
        results_copy = ResultSet()
        results_copy.add_results(self._results)
        return results_copy

    def __str__(self):
        """
        To string dunder method

        Returns
        -------
        str
            A user-friendly representation of the ResultSet
        """
        return str(self._results)

    def __repr__(self):
        """
        String representation dunder method

        Returns
        -------
        str
            A representation of the ResultSet
        """
        return "ResultSet(\n" + str(self) + "\n)"

class RunResultSet(ResultSet):
    """
    Class for storing and managing many RunResult objects
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for RunResultSet
        """
        super().__init__(*args, **kwargs)

    def combine(self):
        """
        Returns an ExperimentResult object with corresponding run DataFrames concatenated

        Returns
        -------
        ExperimentResult
            The results of the experiment (corresponding results from each run merged into a DataFrame)
        """
        combined = ExperimentResult()
        temp = {}
        # Loop over frames and add each and its label to corresponding list in temp dictionary
        for run_key, result in self._results.items():
            for frame_key in result.get_dataframe_names():
                if frame_key in temp:
                    temp.get(frame_key).get("frames").append(result.get_dataframe(frame_key))
                    temp.get(frame_key).get("labels").append(run_key)
                else:
                    temp[frame_key] = {
                        "frames" : [result.get_dataframe(frame_key)],
                        "labels" : [run_key]
                    }
        # Loop over each frame type, concatenate frame list by matching cols, and label using keys
        for frame_key, data in temp.items():
            combined.add_dataframe(
                pd.concat(
                    data.get("frames"), keys=data.get("labels"), names=["run", "index"]
                ), frame_key
            )
        return combined

    def copy(self):
        """
        Returns deep copy of RunResultSet

        Returns
        -------
        RunResultSet
            A deep copy of the current RunResultSet
        """
        results_copy = RunResultSet()
        results_copy.add_results(self._results)
        return results_copy


class ExperimentResultSet(ResultSet):
    """
    Class for storing and managing many ExperimentResult objects
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for ExperimentResultSet
        """
        super().__init__(*args, **kwargs)

    def copy(self):
        """
        Returns deep copy of ExperimentResultSet

        Returns
        -------
        ExperimentResultSet
            A deep copy of the current ExperimentResultSet
        """
        results_copy = ExperimentResultSet()
        results_copy.add_results(self._results)
        return results_copy