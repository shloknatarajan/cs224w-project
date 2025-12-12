#!/usr/bin/env python
"""Generate training curves from DDI GCN log with external features."""
import re
import matplotlib.pyplot as plt
import numpy as np

# Parse the log file
log_file = "ddi_gcn_all.log"

epochs = []
losses = []
val_hits20 = []
test_hits20 = []
best_vals = []

with open(log_file, 'r') as f:
    for line in f:
        match = re.search(
            r'Epoch (\d+) \| loss ([\d.]+) \| val@20 ([\d.]+) \| test@20 ([\d.]+) \| best ([\d.]+)',
            line
        )
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            val_hits20.append(float(match.group(3)))
            test_hits20.append(float(match.group(4)))
            best_vals.append(float(match.group(5)))

epochs = np.array(epochs)
losses = np.array(losses)
val_hits20 = np.array(val_hits20) * 100  # Convert to percentage
test_hits20 = np.array(test_hits20) * 100
best_vals = np.array(best_vals) * 100

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training Loss
ax1 = axes[0]
ax1.plot(epochs, losses, 'b-', alpha=0.7, linewidth=1)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss (GCN + External Features)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 2000])

# Add smoothed loss line
window = 20
if len(losses) > window:
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax1.plot(epochs[window-1:], smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    ax1.legend()

# Plot 2: Hits@20 (Validation and Test)
ax2 = axes[1]
ax2.plot(epochs, val_hits20, 'b-', alpha=0.5, linewidth=1, label='Validation')
ax2.plot(epochs, test_hits20, 'g-', alpha=0.5, linewidth=1, label='Test')
ax2.plot(epochs, best_vals, 'r-', linewidth=2, label='Best Validation')

# Mark best point
best_idx = np.argmax(best_vals)
best_epoch = epochs[best_idx]
best_val = best_vals[best_idx]
ax2.scatter([best_epoch], [best_val], color='red', s=100, zorder=5, marker='*')
ax2.annotate(f'Best: {best_val:.2f}%\n(ep {best_epoch})',
             xy=(best_epoch, best_val), xytext=(best_epoch+100, best_val-5),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Hits@20 (%)', fontsize=12)
ax2.set_title('Hits@20 Performance (GCN + External Features)', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 2000])
ax2.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.savefig('training_curves.pdf', bbox_inches='tight')
print("Saved training_curves.png and training_curves.pdf")

# Create summary statistics
print("\n" + "="*60)
print("TRAINING SUMMARY: GCN + External Features (All)")
print("="*60)
print(f"Features: Morgan (2048) + PubChem (9) + ChemBERTa (768) + Drug-targets (229)")
print(f"Total feature dim: 3054")
print(f"Epochs: {len(epochs)} (evaluated every 5)")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Min loss: {losses.min():.4f} at epoch {epochs[np.argmin(losses)]}")
print()
print("Best Validation Results:")
print(f"  Hits@20: {best_vals.max():.2f}% at epoch {epochs[np.argmax(best_vals)]}")
print(f"  Test@20 at best val: {test_hits20[np.argmax(best_vals)]:.2f}%")
print()
print("Final Results (epoch 2000):")
print(f"  Val@20: {val_hits20[-1]:.2f}%")
print(f"  Test@20: {test_hits20[-1]:.2f}%")
print("="*60)
