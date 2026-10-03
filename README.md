### Tensor Flattening and Reconstruction

This project implements the conversion:

```text
B × C × H × W → B × (CHW) → B × C × H × W
```

The flattening and reconstruction are implemented **from scratch using explicit loops and index arithmetic**. Functions such as `reshape()`, `flatten()`, `ravel()`, and `view()` are not used in the main implementation.

## Repository Contents

```text
tensor_flat.py
AI_Assignment_3_Report.pdf
README.md
```

* `tensor_flat.py` – Python implementation and experiments
* `AI_Assignment_3_Report.pdf` – Assignment report
* `README.md` – Project documentation

## Indexing Algorithm

For flattening:

```text
k = c × H × W + h × W + w
F[b,k] = I[b,c,h,w]
```

For reconstruction:

```text
c = k // (H × W)
r = k % (H × W)
h = r // W
w = r % W

I_hat[b,c,h,w] = F[b,k]
```

### Manual Example

For:

```text
B = 1, C = 2, H = 2, W = 3
```

For `(c,h,w) = (1,0,2)`:

```text
k = 1 × 2 × 3 + 0 × 3 + 2
  = 8
```

Therefore:

```text
I[0,1,0,2] → F[0,8]
```

The reverse indexing recovers `(c,h,w) = (1,0,2)`.

## Experiments

The experiments use:

```text
B = 2
H = 8
W = 8
C = {1, 3, 8, 16, 32, 64, 128, 256, 500}
```

### Results

|  B |   C |  H |  W | Emax | MAE |
| -: | --: | -: | -: | ---: | --: |
|  2 |   1 |  8 |  8 |  0.0 | 0.0 |
|  2 |   3 |  8 |  8 |  0.0 | 0.0 |
|  2 |   8 |  8 |  8 |  0.0 | 0.0 |
|  2 |  16 |  8 |  8 |  0.0 | 0.0 |
|  2 |  32 |  8 |  8 |  0.0 | 0.0 |
|  2 |  64 |  8 |  8 |  0.0 | 0.0 |
|  2 | 128 |  8 |  8 |  0.0 | 0.0 |
|  2 | 256 |  8 |  8 |  0.0 | 0.0 |
|  2 | 500 |  8 |  8 |  0.0 | 0.0 |

All configurations produced **zero reconstruction error**, confirming that the flattening and reconstruction mappings correctly preserve all tensor values.

## How to Run

Install NumPy:

```bash
pip install numpy
```

Run:

```bash
python tensor_flat.py
```

