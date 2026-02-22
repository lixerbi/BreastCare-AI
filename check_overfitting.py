"""
Script to check for overfitting/underfitting
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_and_validate_data
from src.preprocessing import preprocess_pipeline
from src.train import ModelTrainer
from src.evaluate import evaluate_model


def plot_learning_curves(model, X_train, y_train):
    """
    Plot learning curves to detect overfitting
    
    Learning curves show:
    - Training score vs dataset size
    - Validation score vs dataset size
    
    Interpretation:
    - Gap between curves = overfitting
    - Both curves low = underfitting
    - Curves converge at high score = good fit
    """
    print("\n📊 Generating Learning Curves...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('ROC-AUC Score')
    plt.title('Learning Curves - Overfitting Detection')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.85, 1.01)
    
    # Add interpretation
    gap = train_mean[-1] - val_mean[-1]
    plt.text(0.5, 0.02, f'Gap: {gap:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    print("✅ Saved: learning_curves.png")
    plt.show()
    
    return train_mean, val_mean, gap


def analyze_overfitting(train_score, val_score, test_score):
    """
    Analyze and report overfitting status
    """
    print("\n" + "="*60)
    print("🔍 OVERFITTING/UNDERFITTING ANALYSIS")
    print("="*60)
    
    print(f"\n📊 Scores:")
    print(f"   Training Score:   {train_score:.4f}")
    print(f"   Validation Score: {val_score:.4f}")
    print(f"   Test Score:       {test_score:.4f}")
    
    train_val_gap = train_score - val_score
    train_test_gap = train_score - test_score
    
    print(f"\n📏 Gaps:")
    print(f"   Train-Val Gap:  {train_val_gap:.4f}")
    print(f"   Train-Test Gap: {train_test_gap:.4f}")
    
    print("\n🎯 Diagnosis:")
    
    # Check overfitting
    if train_val_gap > 0.10:
        print("   ❌ SEVERE OVERFITTING")
        print("      → Model memorizes training data")
        print("      → Poor generalization")
        print("      → Solutions: Regularization, more data, simpler model")
    elif train_val_gap > 0.05:
        print("   ⚠️  MILD OVERFITTING")
        print("      → Some memorization")
        print("      → Consider: Cross-validation, regularization")
    elif train_val_gap > -0.02:
        print("   ✅ WELL-FITTED (Perfect!)")
        print("      → Excellent generalization")
        print("      → Model is production-ready")
    else:
        print("   🤔 TEST SCORE HIGHER (Unusual but OK)")
        print("      → Test set might be easier")
        print("      → Or excellent generalization")
    
    # Check underfitting
    if train_score < 0.85:
        print("\n   ❌ UNDERFITTING")
        print("      → Model too simple")
        print("      → Solutions: More features, complex model, feature engineering")
    elif train_score < 0.90:
        print("\n   ⚠️  POSSIBLE UNDERFITTING")
        print("      → Model might be too simple")
    else:
        print("\n   ✅ NOT UNDERFITTING")
        print("      → Model captures patterns well")
    
    # Overall verdict
    print("\n" + "="*60)
    if train_val_gap < 0.05 and train_score > 0.90:
        print("🎉 VERDICT: Model is WELL-FITTED and PRODUCTION-READY!")
    elif train_val_gap > 0.10:
        print("⚠️  VERDICT: Address overfitting before deployment")
    elif train_score < 0.85:
        print("⚠️  VERDICT: Model needs improvement (underfitting)")
    else:
        print("✅ VERDICT: Model is acceptable, minor improvements possible")
    print("="*60 + "\n")


def plot_train_vs_test_metrics(train_score, test_score):
    """
    Visual comparison of train vs test scores
    """
    metrics = ['ROC-AUC']
    train_scores = [train_score]
    test_scores = [test_score]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, train_scores, width, label='Training', color='skyblue')
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test', color='lightcoral')
    
    ax.set_ylabel('Score')
    ax.set_title('Training vs Test Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('train_vs_test.png', dpi=300)
    print("✅ Saved: train_vs_test.png")
    plt.show()


def main():
    """Run overfitting analysis"""
    print("🚀 Starting Overfitting/Underfitting Analysis...\n")
    
    # Load data
    print("📂 Loading data...")
    df = load_and_validate_data()
    
    # Preprocess
    print("🔧 Preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df)
    
    # Load trained model
    print("🤖 Loading trained model...")
    trainer = ModelTrainer()
    model = trainer.load_model()
    
    # Get scores
    print("📊 Calculating scores...")
    
    # Training score (from cross-validation)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    train_score = cv_scores.mean()
    
    # Test score
    test_preds = model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score
    test_score = roc_auc_score(y_test, test_preds)
    
    # Validation score (same as train in CV)
    val_score = train_score
    
    # Analyze
    analyze_overfitting(train_score, val_score, test_score)
    
    # Plot learning curves
    train_curve, val_curve, gap = plot_learning_curves(model, X_train, y_train)
    
    # Plot train vs test
    plot_train_vs_test_metrics(train_score, test_score)
    
    print("\n✅ Analysis complete! Check the generated plots.")
    print("   📊 learning_curves.png")
    print("   📊 train_vs_test.png")


if __name__ == "__main__":
    main()