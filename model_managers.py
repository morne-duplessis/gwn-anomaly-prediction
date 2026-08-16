from utils.results import RunResult
from utils.manager import STModelManager
import torch
import pandas as pd

from utils.utils import append_run_to_csv, export_runs_to_csv

class classificationManager(STModelManager):

    def __init__(self):
        super().__init__()

    def _run_single_split(self, config):
        """ The original pipeline logic for a single train/valid/test split. """
        training_params = config.get_training_params()
        # Create a copy of params for train_model, removing keys it doesn't expect
        train_model_params = training_params.copy()
        train_model_params.pop('param_names', None)
        train_loader, valid_loader, test_loader, train_scaler, test_scaler = self.preprocess(**config.get_preprocessing_params())
        train_results = self.train_model(train_loader, valid_loader, train_scaler, **train_model_params)
        test_results = self.test_model(test_loader, test_scaler, **config.get_testing_params())
        
        result = RunResult({**train_results, **test_results})
        append_run_to_csv(result, config, training_params.get('param_names'))
        return result


    def _run_walk_forward(self, config):
        """
        Executes a true walk-forward validation, training a single model instance sequentially.
        """
        train_params = config.get_training_params()
        
        fold_generator, _, test_loader, train_scaler, test_scaler = self.preprocess(**config.get_preprocessing_params())
        
        model = self.model
        model.to(train_params['args']['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['args']['lr'])

        all_fold_valid_metrics = []

        for i, (train_fold_loader, valid_fold_loader) in enumerate(fold_generator):
            print(f"\n--- Starting Walk-Forward Fold {i+1} ---")
            
            fold_results = self.train_model(
                train_fold_loader, valid_fold_loader, train_scaler, 
                **train_params, model=model, optimizer=optimizer
            )
            
            if fold_results and not fold_results['valid'].empty:
                last_metrics = fold_results['valid'].iloc[-1].to_dict()
                all_fold_valid_metrics.append(last_metrics)
                print(f"Fold {i+2} Final F1: {last_metrics.get('f1', 0.0):.4f}")

        if not all_fold_valid_metrics:
            return RunResult({})
            
        agg_metrics_df = pd.DataFrame(all_fold_valid_metrics)
        print("\n--- Walk-Forward Validation Summary (Progress Reports) ---")
        print(agg_metrics_df)
        
        print("\n--- Final Test on Hold-out Set ---")
        test_results = self.test_model(test_loader, test_scaler, **config.get_testing_params())

        result = RunResult({"valid": agg_metrics_df, **test_results})
        append_run_to_csv(result, config, train_params.get('param_names'))
        return result

    def run_pipeline(self, config):
        """
        Executes the machine learning pipeline.
        Checks for a 'walk_forward' flag in the config to decide which validation strategy to use.
        """
        # from torchinfo import summary
        
        # model = self.model
        # args = config.get_training_params().get('args', {})
        # model.to(args['device'])
        
        # Print architecture
        # summary(model, input_size=(1, args['node_cnt'], 1, args['window_size']))
        
        args = config.get_training_params().get('args', {})
        if args.get('walk_forward', False):
            return self._run_walk_forward(config)
        else:
            return self._run_single_split(config)

    from utils.preprocess import preprocess
    from utils.train import train_model
    from utils.validation import validate_model

    def test_model(self, test_loader, test_scaler, args, result_train_file, progress_callback=None):
        metrics = self.validate_model(test_loader, device=args.get("device"), args=args)
        print(
            f"{args.get('model', 'Model')} Test Performance: loss: {metrics['loss']:.4f} | "
            f"avg_acc: {metrics['acc']:.4f} | avg_prec: {metrics['precision']:.4f} | "
            f"avg_recall: {metrics['recall']:.4f} | avg_f1: {metrics['f1']:.4f} | "
            f"avg_onset_prec: {metrics['avg_onset_prec']:.4f} | avg_onset_rec: {metrics['avg_onset_rec']:.4f} | "
            f"avg_onset_f1: {metrics['avg_onset_f1']:.4f} | avg_onset_support: {metrics['avg_onset_support']}"
        )
        test_df = pd.DataFrame([metrics])
        return {"test": test_df}

