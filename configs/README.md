# configs/

One YAML file describes one run; the file plus a git commit is everything needed to reproduce it (spec 014). Load with `adl_etc.utils.config.load_config`.

Existing: `data/` (PCAP flow-builder settings), `splits/` (train/val/test definitions, spec 004). Model, training and evaluation configs are added by the tasks that need them (`models/`, `train/`, `eval/`).

## Composition

A file may start with a `defaults:` list of paths, **relative to the file that names it**. They are merged in order (later wins), then the file's own keys are merged over the result. `defaults` never appears in the resolved config.

```yaml
# configs/train/b3_gru.yaml
defaults: [../models/baselines/gru.yaml, base.yaml]
optim:
  lr: 0.003        # overrides base.yaml's optim.lr; the rest of optim is inherited
```

CLI overrides are dotted `key=value` and win over everything:

```python
cfg = load_config("configs/train/b3_gru.yaml", ["seed=1", "optim.lr=0.002"])
```

An override naming a key that isn't already in the composed config raises. That is deliberate: `optim.learning_rate=0.1` against a config that has `optim.lr` must fail, not silently add a key nothing reads.

The returned config is read-only and fully resolved (`${...}` interpolations substituted). `config_hash(cfg)` hashes the resolved values, so key order, comments and whitespace don't change it.

## Gotchas

- **YAML 1.1 booleans.** A bare `on`, `off`, `yes` or `no` is parsed as `true`/`false`, including as a *key*: `on: true` becomes the key `True`. Use `enabled:` / `enable_x:` for switches, and quote string values that look like booleans (`'no'`).
- **Ints vs strings.** `x: 1` and `x: '1'` are different configs and hash differently.
- Interpolations (`${a.b}`) resolve against the final merged config, so a child can override `a.b` and everything interpolating it follows.
