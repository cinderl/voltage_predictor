import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

try:
    import shap
except Exception:
    shap = None

try:
    from captum.attr import IntegratedGradients
except Exception:
    IntegratedGradients = None

from train_model import TimeValuePredictor, FILE_NAME, MODEL_PATH, STATS_PATH, load_and_preprocess_data, DEVICE

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos']


def load_model(input_size):
    model = TimeValuePredictor(input_size=input_size)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f"[INFO] Loaded model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model.eval()
    return model


def model_predict_numpy(model, X_np):
    # X_np: (n_samples, n_features) numpy
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_np.astype(np.float32))
        out = model(X_tensor).cpu().numpy().reshape(-1)
    return out


def permutation_importance(model, X_val, Y_val, n_repeats=10):
    baseline_preds = model_predict_numpy(model, X_val)
    baseline_mse = mean_squared_error(Y_val.reshape(-1), baseline_preds)
    importances = []
    rng = np.random.RandomState(0)
    for col in range(X_val.shape[1]):
        mses = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            rng.shuffle(X_perm[:, col])
            preds = model_predict_numpy(model, X_perm)
            mses.append(mean_squared_error(Y_val.reshape(-1), preds))
        importances.append(np.mean(mses) - baseline_mse)
    return np.array(importances), baseline_mse


def compute_integrated_gradients(model, X_ref, X_target, device='cpu'):
    if IntegratedGradients is None:
        print('[WARN] Captum not installed; skipping Integrated Gradients')
        return None
    model.to(device)
    ig = IntegratedGradients(model)
    # Use a small batch due to compute cost
    X_ref_t = torch.tensor(X_ref, dtype=torch.float32, device=device)
    X_target_t = torch.tensor(X_target, dtype=torch.float32, device=device)
    attributions = []
    with torch.no_grad():
        for i in range(X_target_t.shape[0]):
            attr = ig.attribute(X_target_t[i:i+1], baselines=X_ref_t.mean(dim=0, keepdim=True), target=0, n_steps=50)
            attributions.append(attr.cpu().numpy().reshape(-1))
    attributions = np.stack(attributions, axis=0)
    return attributions


def run_explanations(sample_count=200, shap_samples=100):
    print('[INFO] Loading preprocessed data (this will reuse the same preprocessing as training)')
    X_train, X_val, Y_train, Y_val = load_and_preprocess_data(FILE_NAME)
    input_size = X_train.shape[1]
    model = load_model(input_size)

    # Permutation importance
    print('[INFO] Computing permutation importance...')
    imp, baseline = permutation_importance(model, X_val, Y_val, n_repeats=20)
    ranked_idx = np.argsort(-imp)
    print('[RESULT] Permutation importance (higher = more important, increase in MSE):')
    for i in ranked_idx:
        print(f"  {FEATURE_NAMES[i]:<12}: {imp[i]:.6f}")

    # Integrated Gradients
    ig_results = None
    if IntegratedGradients is not None:
        print('[INFO] Computing Integrated Gradients on a small subset...')
        # Use a small reference and subset of validation
        ref = torch.tensor(X_train[np.random.choice(len(X_train), min(50, len(X_train))), :], dtype=torch.float32)
        subset_idx = np.random.choice(len(X_val), min(30, len(X_val)), replace=False)
        ig_attributions = compute_integrated_gradients(model, ref, X_val[subset_idx], device='cpu')
        if ig_attributions is not None:
            mean_abs = np.mean(np.abs(ig_attributions), axis=0)
            ranked = np.argsort(-mean_abs)
            print('[RESULT] Integrated Gradients mean absolute attributions:')
            for i in ranked:
                print(f"  {FEATURE_NAMES[i]:<12}: {mean_abs[i]:.6f}")
            ig_results = mean_abs

    # SHAP explanations
    if shap is None:
        print('[WARN] SHAP not installed; skipping SHAP explanations')
    else:
        print('[INFO] Running SHAP KernelExplainer (may be slow)')
        # Create a small background dataset and an eval subset
        background_idx = np.random.choice(len(X_train), min(sample_count, len(X_train)), replace=False)
        background = X_train[background_idx]
        eval_idx = np.random.choice(len(X_val), min(shap_samples, len(X_val)), replace=False)
        eval_data = X_val[eval_idx]

        def f(X):
            # shap expects a 2D array -> return 1D predictions
            return model_predict_numpy(model, X)

        explainer = shap.KernelExplainer(f, background)
        shap_vals = explainer.shap_values(eval_data, nsamples=200)
        # shap_vals will be (n_samples, n_features)
        shap_summary = np.mean(np.abs(shap_vals), axis=0)
        ranked = np.argsort(-shap_summary)
        print('[RESULT] SHAP mean(|SHAP|) feature importances:')
        for i in ranked:
            print(f"  {FEATURE_NAMES[i]:<12}: {shap_summary[i]:.6f}")

        # Save a SHAP summary plot
        try:
            plt.figure(figsize=(8, 4))
            shap.summary_plot(shap_vals, eval_data, feature_names=FEATURE_NAMES, show=False)
            out_path = os.path.join(MODEL_DIR, 'shap_summary.png')
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f"[INFO] SHAP summary plot saved to {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save SHAP plot: {e}")

        # Save SHAP dependence (scatter) plots per feature
        for i, fname in enumerate(FEATURE_NAMES):
            try:
                plt.figure(figsize=(6, 4))
                # shap.dependence_plot accepts either feature name or index
                shap.dependence_plot(i, shap_vals, eval_data, feature_names=FEATURE_NAMES, show=False)
                dep_path = os.path.join(MODEL_DIR, f'shap_dependence_{fname}.png')
                plt.savefig(dep_path, bbox_inches='tight')
                plt.close()
                print(f"[INFO] SHAP dependence plot saved to {dep_path}")
            except Exception as e:
                print(f"[WARN] Could not save SHAP dependence for {fname}: {e}")

    # Save permutation importance plot
    try:
        plt.figure(figsize=(8, 4))
        order = np.argsort(imp)
        plt.barh(np.array(FEATURE_NAMES)[order], imp[order])
        plt.xlabel('Increase in MSE')
        plt.title('Permutation Feature Importance')
        out_path = os.path.join(MODEL_DIR, 'permutation_importance.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Permutation importance plot saved to {out_path}")
    except Exception as e:
        print(f"[WARN] Could not save permutation importance plot: {e}")

    # If IG results exist, save numeric CSV
    if ig_results is not None:
        df = pd.DataFrame({'feature': FEATURE_NAMES, 'ig_mean_abs': ig_results})
        df.sort_values('ig_mean_abs', ascending=False).to_csv(os.path.join(MODEL_DIR, 'integrated_gradients.csv'), index=False)
        print(f"[INFO] Integrated Gradients numeric results saved to {os.path.join(MODEL_DIR, 'integrated_gradients.csv')}")


if __name__ == '__main__':
    print('[INFO] Running model explanation script')
    try:
        run_explanations()
    except Exception as e:
        print(f'[ERROR] {e}')
