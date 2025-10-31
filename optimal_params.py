from a2.cnn import CNNArchConfig, AugmentationConfig, RegularizationConfig
from a2.cnn import evaluate_model
import matplotlib.pyplot as plt

# =============================
# Architecture tuning sweeps
# =============================

print("\n============================")
print("STARTING ARCHITECTURE SWEEPS")
print("============================")

# Augmentation
aug_cfg = AugmentationConfig(
    rotation_deg=0, 
    translate_xy=(0, 0), 
    shear_deg=0
)
# Regularization 
reg_cfg = RegularizationConfig(
    dropout=0.25,          # apply 25% dropout
    weight_decay=1e-4,     # L2 regularization for optimizer
    use_early_stopping=True,
    patience = 3,
    batchnorm=True         # include BatchNorm layers
)

# Convolutional Layer Sweep
depths = [1, 2, 3, 4]
results_depth = []

for d in depths:
    arch_cfg = CNNArchConfig(num_conv_layers=d, num_filters=32, kernel_size=3, pooling="max")
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_depth.append((d, acc))
    print(f"Depth {d}: Val Accuracy = {acc:.4f}")

# Extract best depth
best_depth, best_depth_acc = max(results_depth, key=lambda x: x[1])
print(f"\n Best number of conv layers: {best_depth} (val acc={best_depth_acc:.4f})")

# Plot
plt.figure()
plt.plot([d for d, _ in results_depth], [acc for _, acc in results_depth], marker='o')
plt.title("Effect of Number of Convolutional Layers on Accuracy")
plt.xlabel("Number of Conv Layers")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Filter Sweep
filters_list = [8, 16, 32, 64]
results_filters = []

for f in filters_list:
    arch_cfg = CNNArchConfig(num_conv_layers=best_depth, num_filters=f, kernel_size=3, pooling="max")
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_filters.append((f, acc))
    print(f"Filters {f}: Val Accuracy = {acc:.4f}")

# Extract best filter count
best_filters, best_filter_acc = max(results_filters, key=lambda x: x[1])
print(f"\n Best number of filters: {best_filters} (val acc={best_filter_acc:.4f})")

# Plot
plt.figure()
plt.plot([f for f, _ in results_filters], [acc for _, acc in results_filters], marker='o')
plt.title("Effect of Number of Filters on Accuracy")
plt.xlabel("Number of Filters per Layer")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Kernel Size Sweep
kernels = [3, 5]
results_kernels = []

for k in kernels:
    arch_cfg = CNNArchConfig(num_conv_layers=best_depth, num_filters=best_filters, kernel_size=k, pooling="max")
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_kernels.append((k, acc))
    print(f"Kernel Size {k}: Val Accuracy = {acc:.4f}")

# Extract best kernel size
best_kernel, best_kernel_acc = max(results_kernels, key=lambda x: x[1])
print(f"\n Best kernel size: {best_kernel} (val acc={best_kernel_acc:.4f})")

# Plot
plt.figure()
plt.plot([k for k, _ in results_kernels], [acc for _, acc in results_kernels], marker='o')
plt.title("Effect of Kernel Size on Accuracy")
plt.xlabel("Kernel Size")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()


# Pooling Type Sweep
poolings = ["max", "avg"]
results_poolings = []

for p in poolings:
    arch_cfg = CNNArchConfig(num_conv_layers=best_depth, num_filters=best_filters, kernel_size=best_kernel, pooling=p)
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_poolings.append((p, acc))
    print(f"Pooling {p}: Val Accuracy = {acc:.4f}")

# Extract best pooling
best_pooling, best_pool_acc = max(results_poolings, key=lambda x: x[1])
print(f"\n Best pooling type: {best_pooling} (val acc={best_pool_acc:.4f})")

# Plot
plt.figure()
plt.bar([p for p, _ in results_poolings], [acc for _, acc in results_poolings])
plt.title("Effect of Pooling Type on Accuracy")
plt.xlabel("Pooling Type")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Final optimal architecture
print("\n============================")
print("🏆 FINAL BEST ARCHITECTURE:")
print(f"Conv Layers: {best_depth}")
print(f"Filters per layer: {best_filters}")
print(f"Kernel size: {best_kernel}")
print(f"Pooling type: {best_pooling}")
print("============================")


# =============================
# Augmentation tuning sweeps
# =============================

print("\n============================")
print("STARTING AUGMENTATION SWEEPS")
print("============================")

# Use best architecture from previous sweeps
arch_cfg = CNNArchConfig(
    num_conv_layers=best_depth,
    num_filters=best_filters,
    kernel_size=best_kernel,
    pooling=best_pooling
)

# Rotation Sweep
rotation_values = [0, 5, 10, 20]
results_rotation = []

for deg in rotation_values:
    aug_cfg = AugmentationConfig(rotation_deg=deg, translate_xy=(0, 0), shear_deg=0)
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=3)
    results_rotation.append((deg, acc))
    print(f"Rotation {deg}°: Val Accuracy = {acc:.4f}")

# Find best
best_rotation, best_rot_acc = max(results_rotation, key=lambda x: x[1])
print(f"\n Best rotation degree: {best_rotation}° (val acc={best_rot_acc:.4f})")

# Plot
plt.figure()
plt.plot([r for r, _ in results_rotation], [acc for _, acc in results_rotation], marker='o')
plt.title("Effect of Rotation Degree on Validation Accuracy")
plt.xlabel("Rotation (degrees)")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Translation Sweep
translation_values = [(0, 0), (0.05, 0.05), (0.1, 0.1), (0.2, 0.2)]
results_translation = []

for t in translation_values:
    aug_cfg = AugmentationConfig(rotation_deg=best_rotation, translate_xy=t, shear_deg=0)
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=3)
    results_translation.append((t, acc))
    print(f"Translation {t}: Val Accuracy = {acc:.4f}")

# Find best
best_translation, best_trans_acc = max(results_translation, key=lambda x: x[1])
print(f"\n Best translation: {best_translation} (val acc={best_trans_acc:.4f})")

# Plot
plt.figure()
plt.plot(
    [t[0] for t, _ in results_translation],
    [acc for _, acc in results_translation],
    marker='o'
)
plt.title("Effect of Translation on Validation Accuracy")
plt.xlabel("Translation (fraction of image)")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Shear Sweep
shear_values = [0, 5, 10, 15]
results_shear = []

for s in shear_values:
    aug_cfg = AugmentationConfig(rotation_deg=best_rotation, translate_xy=best_translation, shear_deg=s)
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=3)
    results_shear.append((s, acc))
    print(f"Shear {s}°: Val Accuracy = {acc:.4f}")

# Find best
best_shear, best_shear_acc = max(results_shear, key=lambda x: x[1])
print(f"\n Best shear degree: {best_shear}° (val acc={best_shear_acc:.4f})")

# Plot
plt.figure()
plt.plot([s for s, _ in results_shear], [acc for _, acc in results_shear], marker='o')
plt.title("Effect of Shear on Validation Accuracy")
plt.xlabel("Shear (degrees)")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Final Best Augmentation Summary
print("\n============================")
print("🏆 FINAL BEST AUGMENTATION PARAMETERS:")
print(f"Rotation: {best_rotation}°")
print(f"Translation: {best_translation}")
print(f"Shear: {best_shear}°")
print("============================")


# =============================
# Regularization tuning sweeps
# =============================

print("\n============================")
print("STARTING REGULARIZATION SWEEPS")
print("============================")

# Use best augmentation from previous sweeps
aug_cfg = AugmentationConfig(
    rotation_deg=best_rotation,
    translate_xy=best_translation,
    shear_deg=best_shear
)

# Dropout Sweep
dropout_values = [0.0, 0.25, 0.5]
results_dropout = []

for d in dropout_values:
    reg_cfg = RegularizationConfig(
        dropout=d,
        weight_decay=1e-4,         
        use_early_stopping=True,   
        patience=3,                 
        batchnorm=True              
    )
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_dropout.append((d, acc))
    print(f"Dropout {d}: Val Accuracy = {acc:.4f}")

# Extract best dropout
best_dropout, best_dropout_acc = max(results_dropout, key=lambda x: x[1])
print(f"\n Best Dropout: {best_dropout} (val acc={best_dropout_acc:.4f})")

# Plot Dropout Sweep
plt.figure()
plt.plot([d for d, _ in results_dropout], [acc for _, acc in results_dropout], marker='o')
plt.title("Effect of Dropout on Validation Accuracy")
plt.xlabel("Dropout Rate")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Weight Decay Sweep
weight_decay_values = [0.0, 1e-4, 1e-3]
results_weight_decay = []

for wd in weight_decay_values:
    reg_cfg = RegularizationConfig(
        dropout=best_dropout,
        weight_decay=wd,
        use_early_stopping=True,
        patience=3,
        batchnorm=True
    )
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_weight_decay.append((wd, acc))
    print(f"Weight Decay {wd}: Val Accuracy = {acc:.4f}")

# Extract best weight decay
best_weight_decay, best_wd_acc = max(results_weight_decay, key=lambda x: x[1])
print(f"\n Best Weight Decay: {best_weight_decay} (val acc={best_wd_acc:.4f})")

# Plot Weight Decay Sweep
plt.figure()
plt.semilogx([wd for wd, _ in results_weight_decay], [acc for _, acc in results_weight_decay], marker='o')
plt.title("Effect of Weight Decay on Validation Accuracy")
plt.xlabel("Weight Decay (log scale)")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# BatchNorm Sweep
batchnorm_values = [True, False]
results_batchnorm = []

for bn in batchnorm_values:
    reg_cfg = RegularizationConfig(
        dropout=best_dropout,
        weight_decay=best_weight_decay,
        use_early_stopping=True,
        patience=3,
        batchnorm=bn
    )
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_batchnorm.append((bn, acc))
    print(f"BatchNorm {bn}: Val Accuracy = {acc:.4f}")

# Extract best batchnorm setting
best_bn, best_bn_acc = max(results_batchnorm, key=lambda x: x[1])
print(f"\n Best BatchNorm Setting: {best_bn} (val acc={best_bn_acc:.4f})")

# Plot BatchNorm Sweep
plt.figure()
plt.bar(["BatchNorm ON", "BatchNorm OFF"], [acc for _, acc in results_batchnorm])
plt.title("Effect of Batch Normalization on Validation Accuracy")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()

# Early Stopping Sweep
early_stopping_values = [True, False]
results_earlystop = []

for es in early_stopping_values:
    reg_cfg = RegularizationConfig(
        dropout=best_dropout,
        weight_decay=best_weight_decay,
        use_early_stopping=es,
        patience=3,      
        batchnorm=best_bn
    )
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_earlystop.append(("ON" if es else "OFF", acc))
    print(f"Early Stopping {es}: Val Accuracy = {acc:.4f}")

# Extract best early stopping setting
best_es, best_es_acc = max(results_earlystop, key=lambda x: x[1])
print(f"\n Best Early Stopping: {best_es} (val acc={best_es_acc:.4f})")

# Plot Early Stopping Sweep
plt.figure()
plt.bar([p for p, _ in results_earlystop], [acc for _, acc in results_earlystop])
plt.title("Effect of Early Stopping on Validation Accuracy (Patience=3)")
plt.xlabel("Early Stopping Setting")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()


# RECAP OF ALL OPTIMAL PARAMETERS
# architecture
print("\n============================")
print("🏆 FINAL BEST ARCHITECTURE:")
print(f"Conv Layers: {best_depth}")
print(f"Filters per layer: {best_filters}")
print(f"Kernel size: {best_kernel}")
print(f"Pooling type: {best_pooling}")
print("============================")

# augmentation 
print("\n============================")
print("🏆 FINAL BEST AUGMENTATION PARAMETERS:")
print(f"Rotation: {best_rotation}°")
print(f"Translation: {best_translation}")
print(f"Shear: {best_shear}°")
print("============================")

# regularization
print("\n============================")
print("🏆 FINAL BEST REGULARIZATION PARAMETERS:")
print(f"Dropout: {best_dropout}")
print(f"Weight Decay: {best_weight_decay}")
print(f"BatchNorm: {best_bn}")
print(f"Early Stopping: {best_es}")
print("============================")
