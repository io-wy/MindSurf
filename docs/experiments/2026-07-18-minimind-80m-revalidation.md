# MiniMind 80M controlled revalidation

Status: **completed with concerns**

This document publishes the verified outcome of the legacy MiniMind 80M
pretraining study without importing that repository tree into MindSurf. It is
an evidence archive, not a production-model promotion.

The source result is frozen at legacy commit
`cb887abd3e8927425a9346eaec4bf4f60c17ea5d`. Runtime checkpoints, raw
evaluation samples, and datasets are intentionally not part of this commit.
The machine-readable companion is
[`2026-07-18-minimind-80m-revalidation.json`](2026-07-18-minimind-80m-revalidation.json).

## Decision

The best reproducible Pareto point on the original strict dataset is:

- MHA with hidden size 768, 8 layers, 8 attention heads, 8 KV heads, and
  FFN 3584;
- from-scratch training with WSD, learning rate `5e-4`, batch size 32,
  sequence length 384, 10,000 optimizer steps, and 122,880,000 seen tokens;
- stage-1 continuation on quality70/English15/math15 for 700 steps at
  sequence length 512, batch size 16, and learning rate `8e-7`;
- stage-2 continuation on the same mix for 1,200 steps at learning rate
  `3e-7`;
- independent confirmation with seeds 42 and 7.

This recipe is **not approved for infrastructure promotion**. Every final
candidate failed all six frozen domain-loss checks and the MCQ minimum. Work
on release infrastructure must remain gated until a new dataset receives an
independent manifest, leakage audit, training run, and evaluation.

## Immutable data contract

| Artifact | SHA-256 |
| --- | --- |
| Strict manifest | `64e18821cc7c18d96fd149a0480d774fcfd6d30564b96d072a7d8b1d5726270f` |
| Strict train | `a121ae89a213e8223bc8bf7620a031344d1a348627a727f87f56135957d5edaf` |
| Strict validation | `664772d3a8420fe49104c6c531c7eebeeaf92290dd4ab3d5bb34fe88a2fc1f34` |
| Strict test | `a09ece1f82cf059b570f0ed27a758c87929bbd04d9518bf118d118b23cfb4daa` |
| Tokenizer bundle | `7e76729752b4463589393a037bd6a6b1ff370964de1870216dec07a3f1d273ff` |
| Candidate thresholds | `d975c2b1dac4b7021d97efcc0b56fb9ed96b1260faa7b922b61ec45f787d8d5b` |

The strict train contained 1,265,983 rows. Validation and test contained
2,000 rows each. Normalized-text hashes reserved by validation or test were
excluded from training.

The frozen gate required:

- strict validation loss at most `2.42` and strict test loss at most `2.44`;
- MCQ accuracy at least `23/48`;
- fixed-prompt score at least `0.4575`;
- high-repetition count at most 4;
- all six domain losses at or below their historical maxima.

MCQ confidence intervals used 10,000 bootstrap samples with seed `20260712`.

## From-scratch funnel

Every launched arm used the same data identities, tokenizer, data order, WSD
schedule, batch size 32, sequence length 384, 10,000 optimizer steps, and
122,880,000 seen tokens.

| Run | Strict val | Strict test | Mean | Tokens/s | Peak GiB | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MHA/FFN3072, LR `3e-4` | `2.438767` | `2.452744` | `2.445755` | `104132.66` | `7.947` | Rejected: strict regression |
| MHA/FFN3072, LR `5e-4` | `2.417287` | `2.434266` | `2.425776` | `104245.35` | `7.947` | LR reference |
| MHA/FFN3072, LR `7e-4` | `2.410559` | `2.427583` | `2.419071` | `104237.10` | `7.947` | Practical tie |
| GQA/FFN3328, LR `5e-4` | `2.428264` | `2.441455` | `2.434859` | `110189.39` | `8.004` | Rejected: validation regression |
| MHA/FFN2816, LR `5e-4` | `2.430995` | `2.443201` | `2.437098` | `107355.21` | `8.014` | Rejected: strict regression |
| MHA/FFN3584, LR `5e-4` | `2.399181` | `2.415155` | `2.407168` | `98841.02` | `8.811` | Selected parent |
| MHA/FFN4096, LR `5e-4` | `2.397700` | `2.411336` | `2.404518` | `94004.80` | `8.943` | Practical tie; less efficient |

FFN3584 has 89,864,448 parameters. FFN4096 has 99,301,632 parameters and
improved strict mean by only `0.002650`, below the predefined `0.01`
practical-significance trigger. The FFN4608 arm was therefore not run.

FFN3584 was selected over FFN4096 because it had higher MCQ accuracy, lower
loss in five of six diagnostic domains, 5.1% higher throughput, and 9,437,184
fewer parameters.

## Final replay control

For each seed, the quality70 control and quality80 replay arms started from
the exact same quality70 stage-1 parent. Each stage-2 arm used 1,200 steps,
19,200 consumed blocks, sequence length 512, batch size 16, learning rate
`3e-7`, and 9,830,400 seen tokens.

| Seed / arm | Strict val | Strict test | Mean | MCQ | Fixed | Repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 / quality70 control | `2.391499` | `2.409126` | `2.400313` | `15/48` | `0.44` | `2` |
| 42 / quality80 replay | `2.390968` | `2.408508` | `2.399738` | `15/48` | `0.44` | `2` |
| 7 / quality70 control | `2.396327` | `2.412125` | `2.404226` | `17/48` | `0.475` | `1` |
| 7 / quality80 replay | `2.395817` | `2.411612` | `2.403715` | `17/48` | `0.3785` | `3` |

Replay improved strict mean by only `0.000574` at seed 42 and `0.000512` at
seed 7. MCQ accuracy did not change. At seed 7, replay reduced the fixed
prompt score from `0.475` to `0.3785` and increased high-repetition cases from
1 to 3. The Pareto tie therefore resolves in favor of the quality70 control.

## Interpretation boundaries

- The selected recipe is the best reproducible point on the original dataset,
  not a promoted model.
- Results from a replacement dataset must use new run names and immutable
  identities; they must not be combined with these tables.
- Loss improvements inside the `0.01` practical-tie band are not treated as
  capability gains.
- The legacy implementation is not copied into this branch because its
  repository layout and runtime contracts differ from the current MindSurf
  tree. Any future integration must be reimplemented against MindSurf's
  Hydra, DVC, trainer, and evaluation interfaces.
