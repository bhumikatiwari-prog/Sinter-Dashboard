import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, pickle, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb, lightgbm as lgb, catboost as cb
import shap
from feature_engineering import engineer, LEAKAGE_SQ, LEAKAGE_PP, TARGETS

ARTIFACTS = os.path.join(os.path.dirname(__file__), 'artifacts')

def load_clean_data(path):
    pp = pd.read_excel(path, sheet_name='Process_Parameters_Clean')
    sq = pd.read_excel(path, sheet_name='Sinter_Quality_Clean')
    for c in pp.columns:
        if c != 'Date': pp[c] = pd.to_numeric(pp[c], errors='coerce')
    for c in sq.columns:
        if c != 'DATE': sq[c] = pd.to_numeric(sq[c], errors='coerce')
    sq = sq[sq['RDI'] < 35].copy() if 'RDI' in sq.columns else sq
    pp.drop(columns=[c for c in LEAKAGE_PP if c in pp.columns], inplace=True, errors='ignore')
    sq.drop(columns=[c for c in LEAKAGE_SQ if c in sq.columns], inplace=True, errors='ignore')
    return pp, sq

def prep_xy(df, target, thresh=0.50):
    others = [t for t in TARGETS if t != target]
    drop = [c for c in ['DATE','Date']+others if c in df.columns]
    y = df[target].dropna()
    X = df.loc[y.index].drop(columns=drop+[target], errors='ignore')
    X = X.dropna(axis=1, thresh=int((1-thresh)*len(X)))
    return X, y

def train(data_path, artifacts_dir=ARTIFACTS):
    os.makedirs(artifacts_dir, exist_ok=True)
    pp, sq = load_clean_data(data_path)
    sq_fe  = engineer(sq, is_pp=False)
    pp_fe  = engineer(pp, is_pp=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_models, all_results, shap_data = {}, {}, {}

    CANDIDATES = {
        'RandomForest': RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=3,random_state=42,n_jobs=-1),
        'XGBoost':      xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.04,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.1,random_state=42,n_jobs=-1,verbosity=0),
        'LightGBM':     lgb.LGBMRegressor(n_estimators=400,max_depth=6,learning_rate=0.04,num_leaves=35,subsample=0.8,colsample_bytree=0.75,min_child_samples=10,random_state=42,n_jobs=-1,verbose=-1),
        'CatBoost':     cb.CatBoostRegressor(iterations=400,depth=6,learning_rate=0.04,l2_leaf_reg=3.0,random_seed=42,verbose=0),
    }

    for target in TARGETS:
        X, y = prep_xy(sq_fe, target)
        imp  = SimpleImputer(strategy='median')
        Xi   = imp.fit_transform(X); Xdf = pd.DataFrame(Xi, columns=X.columns)
        results = {}
        for name, model in CANDIDATES.items():
            yp = cross_val_predict(model, Xdf, y, cv=kf, n_jobs=-1)
            results[name] = {
                'R2':   round(r2_score(y,yp),4),
                'RMSE': round(np.sqrt(mean_squared_error(y,yp)),4),
                'MAE':  round(mean_absolute_error(y,yp),4),
                'MAPE': round(np.mean(np.abs((y.values-yp)/y.values.clip(min=1e-6)))*100,3),
            }
        best_name = max(results, key=lambda m: results[m]['R2'])
        best_m = CANDIDATES[best_name]; best_m.fit(Xdf, y)

        # SHAP
        try:
            exp = shap.TreeExplainer(best_m)
            sv  = exp.shap_values(Xdf)
            if isinstance(sv, list): sv = sv[0]
        except:
            sv = np.zeros((len(Xdf), len(X.columns)))
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:15]

        all_results[target]  = results
        best_models[target]  = {'model':best_m,'imputer':imp,'features':X.columns.tolist(),
                                 'best_algo':best_name,'metrics':results[best_name]}
        shap_data[target]    = {'sv':sv,'feats':X.columns.tolist(),'X':Xdf,
                                 'top_feats':[X.columns[i] for i in top_idx],
                                 'top_vals':mean_abs[top_idx]}

    # Feature importance from best model per target
    feat_imp = {}
    for target in TARGETS:
        info = best_models[target]
        m = info['model']
        if hasattr(m,'feature_importances_'):
            fi = pd.Series(m.feature_importances_, index=info['features']).nlargest(15)
        else:
            fi = pd.Series(shap_data[target]['top_vals'], index=shap_data[target]['top_feats'])
        feat_imp[target] = fi

    # Correlation matrix
    corr_cols = [c for c in sq_fe.columns if c not in ['DATE','Date']]
    corr = sq_fe[corr_cols].corr()

    artifacts = {
        'best_models':  best_models,
        'all_results':  all_results,
        'shap_data':    shap_data,
        'feat_imp':     feat_imp,
        'sq_fe':        sq_fe,
        'pp_fe':        pp_fe,
        'sq_raw':       sq,
        'pp_raw':       pp,
        'corr':         corr,
    }
    with open(os.path.join(artifacts_dir,'artifacts.pkl'),'wb') as f:
        pickle.dump(artifacts, f)
    print(f"[✓] Training complete. Artifacts saved to {artifacts_dir}/artifacts.pkl")
    for t in TARGETS:
        r = all_results[t][best_models[t]['best_algo']]
        print(f"  {t}: {best_models[t]['best_algo']}  R²={r['R2']}  RMSE={r['RMSE']}  MAPE={r['MAPE']}%")
    return artifacts

if __name__ == '__main__':
    train('/mnt/user-data/uploads/Outlier_Removed_Data.xlsx')
