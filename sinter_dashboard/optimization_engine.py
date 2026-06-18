import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from prediction_engine import predict_from_row, build_input_row

TARGETS = ['TI','RDI','RI']

def run_optimization(best_models, sq_raw, n_trials=300):
    levers = ['%FeO','%CaO','%MgO','%SiO2','%Al2O3','Avl. lime','Basicity  (B2)','MgO/Al2O3']
    levers = [c for c in levers if c in sq_raw.columns]
    ranges = sq_raw[levers].quantile([0.05,0.95]).T; ranges.columns=['lo','hi']

    def objective(trial):
        params = {col: trial.suggest_float(col, float(ranges.loc[col,'lo']),
                                                float(ranges.loc[col,'hi']))
                  for col in levers}
        sq_fe_proxy = sq_raw.copy()
        row = build_input_row(params, sq_fe_proxy)
        preds = predict_from_row(row, best_models)
        ti, rdi, ri = preds['TI'], preds['RDI'], preds['RI']
        trial.set_user_attr('TI',  round(ti,3))
        trial.set_user_attr('RDI', round(rdi,3))
        trial.set_user_attr('RI',  round(ri,3))
        return -(ti + ri - rdi * 2.5)

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    pareto = pd.DataFrame([{
        'TI':  t.user_attrs.get('TI',  np.nan),
        'RDI': t.user_attrs.get('RDI', np.nan),
        'RI':  t.user_attrs.get('RI',  np.nan),
        **t.params} for t in study.trials]).dropna()
    return {
        'best_TI':  best.user_attrs['TI'],
        'best_RDI': best.user_attrs['RDI'],
        'best_RI':  best.user_attrs['RI'],
        'best_params': best.params,
        'pareto':   pareto,
        'ranges':   ranges,
    }
