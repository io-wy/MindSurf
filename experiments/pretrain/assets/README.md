# Pretraining Asset Restoration

A Git checkout restores source code, configurations, evaluation suites, strict
split fixtures, summaries, and reports. It does not restore full datasets,
model checkpoints, profiler databases, platform run directories, logs,
wheelhouses, or vendored runtime repositories.

`asset_manifest.json` is the local inventory for externally stored assets. Its
recorded file sizes and SHA-256 hashes are integrity gates, not hints. Keep full
datasets and checkpoints on the training server or approved external storage;
do not add them to Git.

The `server://project/` locator resolves to the configured project root on the
training server.

To rebuild the inventory from a repository checkout, run:

```powershell
python scripts/build_asset_manifest.py --sources experiments/pretrain/assets/asset_sources.json --root . --output experiments/pretrain/assets/asset_manifest.json
```

Restore an asset only to the exact `logical_path` listed by its source
definition. Before changing any manifest entry from `missing` to `present`,
verify that the file exists at that exact path and let the manifest CLI compute
both its byte size and SHA-256. Never hand-edit availability, size, or hash
fields. In particular, the stage12 checkpoint remains `missing` unless its
exact documented path exists.
