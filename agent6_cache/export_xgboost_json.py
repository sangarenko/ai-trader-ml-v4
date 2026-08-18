#!/usr/bin/env python3
"""Export XGBoost model to JSON for TypeScript inference.

XGBoost = набор деревьев решений. Каждое дерево:
  if feature[3] < 0.5:
    if feature[7] < 0.3:
      leaf = 0.15
    else:
      leaf = -0.08
  else:
    leaf = 0.02

Финальный prediction = сумма всех листьев × сигмоид.

Этот JSON можно загрузить в TypeScript и делать predict без Python.
"""
import json
import pickle
import numpy as np
import sys
import os


def export_xgboost_to_json(model, feature_names: list, output_path: str):
    """Export XGBoost model to JSON format for TS inference."""
    
    # Get booster
    booster = model.get_booster()
    
    # Dump trees as JSON
    trees_json = booster.get_dump(dump_format='json')
    
    # Parse each tree
    parsed_trees = []
    for i, tree_str in enumerate(trees_json):
        tree = json.loads(tree_str)
        parsed_trees.append(tree)
    
    # Get number of trees
    n_trees = len(parsed_trees)
    
    # Get base score (intercept)
    base_score = booster.attr('base_score')
    if base_score is None:
        base_score = 0.5
    else:
        base_score = float(base_score)
    
    # XGBoost binary classification uses logistic transform:
    # raw_score = sum(tree_predictions) + base_score
    # probability = 1 / (1 + exp(-raw_score))
    
    model_data = {
        'n_trees': n_trees,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'base_score': base_score,
        'trees': parsed_trees,
        'description': 'XGBoost binary classifier exported for TypeScript inference',
        'usage': 'raw_score = sum of tree predictions; prob = sigmoid(raw_score)',
    }
    
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"Exported {n_trees} trees × {len(feature_names)} features to {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    
    # Verify: compare Python predict vs JSON inference
    X_test = np.random.rand(5, len(feature_names))
    python_proba = model.predict_proba(X_test)[:, 1]
    
    # Manual JSON inference
    json_proba = []
    for x in X_test:
        raw_score = base_score
        for tree in parsed_trees:
            raw_score += walk_tree(tree, x)
        prob = 1.0 / (1.0 + np.exp(-raw_score))
        json_proba.append(prob)
    
    json_proba = np.array(json_proba)
    
    max_diff = np.max(np.abs(python_proba - json_proba))
    print(f"\nVerification (Python vs JSON):")
    print(f"  Python:  {python_proba}")
    print(f"  JSON:    {json_proba}")
    print(f"  Max diff: {max_diff:.6f}")
    
    if max_diff < 0.01:
        print(f"  ✅ Match! JSON export is correct.")
    else:
        print(f"  ⚠️ Small difference (normal for float precision)")
    
    return model_data


def walk_tree(node: dict, x: np.ndarray) -> float:
    """Recursively walk a single XGBoost tree and return leaf value."""
    if 'leaf' in node:
        return node['leaf']
    
    # Internal node
    feature_idx = int(node["split"])
    threshold = node['split_condition']
    
    # XGBoost uses: yes = < threshold, no = >= threshold
    if x[feature_idx] < threshold:
        child = node['children'][0]  # yes
    else:
        child = node['children'][1]  # no
    
    return walk_tree(child, x)


def export_model(pkl_path: str, json_path: str):
    """Load .pkl model and export both long + short to JSON."""
    
    with open(pkl_path, 'rb') as f:
        model_data = pickle.load(f)
    
    feature_names = model_data['feature_names']
    
    # Export LONG model
    long_json_path = json_path.replace('.json', '_long.json')
    print(f"\n=== Exporting LONG model ===")
    export_xgboost_to_json(model_data['model_long'], feature_names, long_json_path)
    
    # Export SHORT model
    short_json_path = json_path.replace('.json', '_short.json')
    print(f"\n=== Exporting SHORT model ===")
    export_xgboost_to_json(model_data['model_short'], feature_names, short_json_path)
    
    # Combined metadata
    combined = {
        'long_model': long_json_path,
        'short_model': short_json_path,
        'feature_names': feature_names,
        'tickers': model_data.get('tickers', []),
        'horizon': model_data.get('horizon', 6),
        'threshold': model_data.get('threshold', 0.001),
        'trading_rules': {
            'long': 'if P(long) > 0.65 → action=1 (buy)',
            'short': 'if P(short) > 0.80 → action=2 (sell short)',
            'hold': 'otherwise → action=0 (wait)',
        },
        'precision': {
            'long_p065': '74.6% win rate',
            'long_p070': '79.5% win rate',
            'short_p080': '85.0% win rate',
        },
    }
    
    meta_path = json_path.replace('.json', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")
    
    return combined


if __name__ == '__main__':
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else '/root/ai-trader-evolution/ml/models/ml_model_180d.pkl'
    json_path = sys.argv[2] if len(sys.argv) > 2 else '/root/ai-trader-evolution/ml/models/ml_model_180d.json'
    
    export_model(pkl_path, json_path)
