# Centering API

This document describes the centering behavior for the unified `Assembly` node.

## Why This Update

Centering may not reach very strict thresholds (for example `1e-5`) when frame estimation depends on fitted parametric models.
To make convergence criteria explicit and tunable, `center()` supports separate tolerances for rotation and translation.

## API

### Assembly.center (居中自身)

```python
def center(
    self,
    max_try=10,
    atol_rot: float = 1e-5,
    atol_trans: float = 1e-5,
    verbose: bool = False,
)
```

Parameters:
- `max_try`: Maximum centering iterations.
- `atol_rot`: Rotation convergence tolerance in radians, checked against Euler xyz residuals.
- `atol_trans`: Translation convergence tolerance, checked against centroid translation residual.
- `verbose`: Print iteration-by-iteration progress.

Convergence rule:
- Rotation residual is within `atol_rot` **and** translation residual is within `atol_trans`.

### 居中一个 Assembly 中的某个子节点

旧 API 曾有 `Assembly.center(part_index)` (居中一个 part 并对其余 part 施加同一刚体变换),
该操作已移除。统一模型下的正确顺序是: 先在子节点上调用 `center()` 居中它自己,
再在父节点调用 `merge_up()` 用子节点结构重建父结构。

```python
child = assembly["helix_1"]
child.center(max_try=30, atol_rot=1e-5, atol_trans=1e-4, verbose=True)
assembly.merge_up()
```

## Recommended Settings

For parametric/fitted models (for example CCCP bundle workflows), start with:

```python
part.center(max_try=30, atol_rot=1e-5, atol_trans=1e-4, verbose=True)
```

If centering is still not converged:
- Increase `max_try` (for example `50`).
- Relax `atol_trans` to `2e-4` or `5e-4` depending on task sensitivity.

## Example

```python
part = CCCPHelixBundle.from_param(helix_num=2)
part.structure = structure
part.mask = {...}

part.center(max_try=30, atol_rot=1e-5, atol_trans=1e-4, verbose=True)
part.fit(verbose=True)
```
