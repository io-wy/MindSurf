# Strict Pretrain Splits Manifest

Updated: 2026-05-25 16:38:11

This folder contains deterministic text-hash based validation and test splits for MiniMind pretraining.
The selection is global over the source corpus after exact normalized-text deduplication, so it is stricter than using only the first 2k or tail 2k lines.

## Files

- validation: `D:\UserData\Desktop\minimind\experiments\pretrain\strict_splits\pretrain_strict_val_2k.jsonl`
- test: `D:\UserData\Desktop\minimind\experiments\pretrain\strict_splits\pretrain_strict_test_2k.jsonl`
- metadata: `D:\UserData\Desktop\minimind\experiments\pretrain\strict_splits\strict_splits_meta.json`

## Counts

- source lines: `1270238`
- unique texts: `1269983`
- duplicate texts skipped: `255`
- val lines: `2000`
- test lines: `2000`

## Platform Command

```bash
python experiments/pretrain/scripts/prepare_strict_splits.py --write-train
```
