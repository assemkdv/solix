# flare-indexer documentation

| If you want to… | Read |
|---|---|
| **Learn to use the package, step by step (main guide)** | **[researcher_manual.md](researcher_manual.md)** |
| **Look up any public class, function or method** | **[api_reference.md](api_reference.md)** |
| **Understand how the code works inside** | **[architecture/system_design.md](architecture/system_design.md)** |
| Label active-region patches | [active_region_workflow.md](active_region_workflow.md) |
| Check the real-data validation | [validation/full_disk.md](validation/full_disk.md), [validation/active_region.md](validation/active_region.md), [validation/cumulative_index.md](validation/cumulative_index.md) |
| See which choices still need a scientific decision | [decisions.md](decisions.md) |

Runnable examples with synthetic data are in [`../examples`](../examples).
Examples 07–08 (PyTorch) and 10–11 (TensorFlow) label real SDO/HMI
images and load them as training batches;
see [`../examples/real_data/README.md`](../examples/real_data/README.md).

Examples in `researcher_manual.md`, `api_reference.md` and
`architecture/system_design.md` run against the synthetic files in
[`fixtures/`](fixtures/); their recorded output is checked by
`python scripts/check_doc_examples.py` (also part of the test suite).
