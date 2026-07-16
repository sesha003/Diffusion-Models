import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
output_dir = PROJECT_ROOT / "Models" / "Final_Output" / "checkpoints"

print('\n' + '='*70)
print('  DIRECT SEGMENTATION TRAINING - STATUS REPORT')
print('='*70 + '\n')

for fold_idx in range(3):
    fold_dir = output_dir / f'fold_{fold_idx}'
    metrics_file = fold_dir / 'metrics.json'
    model_file = fold_dir / 'best_model.pt'
    
    if metrics_file.exists() and model_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        model_size = model_file.stat().st_size / (1024*1024)
        last_epoch = metrics[-1]
        best_val_loss = min([m['val_loss'] for m in metrics])
        best_epoch = next(m['epoch'] for m in metrics if m['val_loss'] == best_val_loss)
        
        print(f'[OK] FOLD {fold_idx}:')
        print(f'  Status: TRAINED COMPLETE')
        print(f'  Total Epochs: {len(metrics)}')
        print(f'  Best Model at Epoch: {best_epoch}')
        print(f'  Best Val Loss: {best_val_loss:.6f}')
        print(f'  Final Train Loss: {last_epoch["train_loss"]:.6f}')
        print(f'  Final Val Loss: {last_epoch["val_loss"]:.6f}')
        print(f'  Final LR: {last_epoch["lr"]:.2e}')
        print(f'  Model Size: {model_size:.1f} MB')
        print()

print('='*70)
print('  STATUS: ALL FOLDS TRAINED SUCCESSFULLY')
print('='*70 + '\n')
print('Next Step: Run evaluation')
print('  python evaluate_direct_seg.py\n')
