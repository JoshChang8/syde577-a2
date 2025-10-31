from cnn.configs import CNNArchConfig, AugmentationConfig, RegularizationConfig
from cnn.train_eval import evaluate_model
import matplotlib.pyplot as plt

# =============================
# Architecture tuning sweeps
# =============================

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

# 1️⃣ Convolutional Layer Sweep
depths = [1, 2, 3, 4]
results_depth = []

for d in depths:
    arch_cfg = CNNArchConfig(num_conv_layers=d, num_filters=32, kernel_size=3, pooling="max")
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_depth.append((d, acc))
    print(f"Depth {d}: Val Accuracy = {acc:.4f}")

# Extract best depth
best_depth, best_depth_acc = max(results_depth, key=lambda x: x[1])
print(f"\n✅ Best number of conv layers: {best_depth} (val acc={best_depth_acc:.4f})")

# Plot
plt.figure()
plt.plot([d for d, _ in results_depth], [acc for _, acc in results_depth], marker='o')
plt.title("Effect of Number of Convolutional Layers on Accuracy")
plt.xlabel("Number of Conv Layers")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()


# 2️⃣ Filter Sweep
filters_list = [8, 16, 32, 64]
results_filters = []

for f in filters_list:
    arch_cfg = CNNArchConfig(num_conv_layers=best_depth, num_filters=f, kernel_size=3, pooling="max")
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_filters.append((f, acc))
    print(f"Filters {f}: Val Accuracy = {acc:.4f}")

# Extract best filter count
best_filters, best_filter_acc = max(results_filters, key=lambda x: x[1])
print(f"\n✅ Best number of filters: {best_filters} (val acc={best_filter_acc:.4f})")

# Plot
plt.figure()
plt.plot([f for f, _ in results_filters], [acc for _, acc in results_filters], marker='o')
plt.title("Effect of Number of Filters on Accuracy")
plt.xlabel("Number of Filters per Layer")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()


# 3️⃣ Kernel Size Sweep
kernels = [3, 5]
results_kernels = []

for k in kernels:
    arch_cfg = CNNArchConfig(num_conv_layers=best_depth, num_filters=best_filters, kernel_size=k, pooling="max")
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_kernels.append((k, acc))
    print(f"Kernel Size {k}: Val Accuracy = {acc:.4f}")

# Extract best kernel size
best_kernel, best_kernel_acc = max(results_kernels, key=lambda x: x[1])
print(f"\n✅ Best kernel size: {best_kernel} (val acc={best_kernel_acc:.4f})")

# Plot
plt.figure()
plt.plot([k for k, _ in results_kernels], [acc for _, acc in results_kernels], marker='o')
plt.title("Effect of Kernel Size on Accuracy")
plt.xlabel("Kernel Size")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()


# 4️⃣ Pooling Type Sweep
poolings = ["max", "avg"]
results_poolings = []

for p in poolings:
    arch_cfg = CNNArchConfig(num_conv_layers=best_depth, num_filters=best_filters, kernel_size=best_kernel, pooling=p)
    acc = evaluate_model(arch_cfg, aug_cfg, reg_cfg, num_epochs=5)
    results_poolings.append((p, acc))
    print(f"Pooling {p}: Val Accuracy = {acc:.4f}")

# Extract best pooling
best_pooling, best_pool_acc = max(results_poolings, key=lambda x: x[1])
print(f"\n✅ Best pooling type: {best_pooling} (val acc={best_pool_acc:.4f})")

# Plot
plt.figure()
plt.bar([p for p, _ in results_poolings], [acc for _, acc in results_poolings])
plt.title("Effect of Pooling Type on Accuracy")
plt.xlabel("Pooling Type")
plt.ylabel("Validation Accuracy")
plt.grid(True)
plt.show()


# 🏁 Final optimal architecture
print("\n============================")
print("🏆 FINAL BEST ARCHITECTURE:")
print(f"Conv Layers: {best_depth}")
print(f"Filters per layer: {best_filters}")
print(f"Kernel size: {best_kernel}")
print(f"Pooling type: {best_pooling}")
print("============================")
