# experiments/quantum_vs_classical_xai.py

"""
Étude comparative: Quantum Kernel vs. QCNN vs. Classical
pour explicabilité sur CLEVR-Hans.
"""

def run_comparative_xai_study(config):
    """
    Compare 3 approches sur CLEVR-Hans:
    1. Quantum Kernel SVM + Kernel SHAP
    2. Hybrid QCNN + GradCAM/IG
    3. Classical CNN + GradCAM/IG (baseline)
    """
    
    # Load data
    train_loader, val_loader, test_loader = get_clevr_hans_loaders(...)
    
    # ═══════════════════════════════════════════════════════
    # MODEL 1: Quantum Kernel SVM
    # ═══════════════════════════════════════════════════════
    print("Training Quantum Kernel SVM...")
    
    # Extract features (PCA)
    X_train, y_train = extract_features_pca(train_loader, n_components=10)
    X_test, y_test = extract_features_pca(test_loader, n_components=10)
    
    # Compute quantum kernel
    K_train = compute_kernel_matrix(X_train, weights=quantum_weights, symmetric=True)
    K_test = compute_kernel_matrix(X_test, Y=X_train, weights=quantum_weights, symmetric=False)
    
    # Train SVM
    svm = SVC(kernel="precomputed", C=1.0)
    svm.fit(K_train, y_train)
    
    # Evaluate
    acc_qkernel = svm.score(K_test, y_test)
    
    # Explainability
    qkernel_shap = QuantumKernelSHAP(svm, K_train, X_train)
    shap_values_qkernel = qkernel_shap.explain(X_test[:100])
    
    # Metrics
    metrics_qkernel = {
        "accuracy": acc_qkernel,
        "feature_importance": np.abs(shap_values_qkernel).mean(axis=0),
        "confounder_detection": analyze_confounder_features(shap_values_qkernel),
    }
    
    # ═══════════════════════════════════════════════════════
    # MODEL 2: Hybrid QCNN
    # ═══════════════════════════════════════════════════════
    print("Training Hybrid QCNN...")
    
    qcnn = CLEVRQCNNClassifier(n_classes=3, n_qubits=10)
    train_qcnn(qcnn, train_loader, val_loader)
    
    # Evaluate
    acc_qcnn = evaluate(qcnn, test_loader)
    
    # Explainability
    grad_explainer = GradientExplainer(qcnn)
    
    qcnn_attributions = []
    for batch in test_loader:
        images = batch["image"]
        gradcam = grad_explainer.gradcam(images)
        ig = grad_explainer.integrated_gradients(images)
        qcnn_attributions.append({"gradcam": gradcam, "ig": ig})
    
    # Metrics
    metrics_qcnn = {
        "accuracy": acc_qcnn,
        "pixel_attribution": analyze_pixel_attributions(qcnn_attributions),
        "confounder_detection": detect_confounders_from_pixels(qcnn_attributions),
    }
    
    # ═══════════════════════════════════════════════════════
    # MODEL 3: Classical CNN (Baseline)
    # ═══════════════════════════════════════════════════════
    print("Training Classical CNN...")
    
    classical_cnn = ClassicalCNN(n_classes=3)
    train_classical(classical_cnn, train_loader, val_loader)
    
    acc_classical = evaluate(classical_cnn, test_loader)
    
    # Explainability (same methods)
    grad_explainer_classical = GradientExplainer(classical_cnn)
    classical_attributions = []
    for batch in test_loader:
        images = batch["image"]
        gradcam = grad_explainer_classical.gradcam(images)
        ig = grad_explainer_classical.integrated_gradients(images)
        classical_attributions.append({"gradcam": gradcam, "ig": ig})
    
    metrics_classical = {
        "accuracy": acc_classical,
        "pixel_attribution": analyze_pixel_attributions(classical_attributions),
        "confounder_detection": detect_confounders_from_pixels(classical_attributions),
    }
    
    # ═══════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("COMPARATIVE RESULTS")
    print("="*60)
    
    comparison = {
        "Quantum Kernel SVM": metrics_qkernel,
        "Hybrid QCNN": metrics_qcnn,
        "Classical CNN": metrics_classical,
    }
    
    # Plot comparison
    plot_xai_comparison(comparison)
    
    # Specific analysis: Confounder detection
    print("\nConfounder Detection Performance:")
    for model_name, metrics in comparison.items():
        print(f"  {model_name}: {metrics['confounder_detection']:.2%}")
    
    # W&B logging
    wandb.log({
        "comparison/qkernel_acc": metrics_qkernel["accuracy"],
        "comparison/qcnn_acc": metrics_qcnn["accuracy"],
        "comparison/classical_acc": metrics_classical["accuracy"],
        "comparison/qkernel_confounder": metrics_qkernel["confounder_detection"],
        "comparison/qcnn_confounder": metrics_qcnn["confounder_detection"],
        "comparison/classical_confounder": metrics_classical["confounder_detection"],
    })
    
    return comparison
