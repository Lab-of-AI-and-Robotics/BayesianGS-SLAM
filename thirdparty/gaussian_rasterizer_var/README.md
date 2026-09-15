# Gaussian Rasterizer with Per-Splat Variance

Modified differential Gaussian rasterizer used in [VarSplat](https://github.com/anhthuan1999/varsplat). Built on top of [gaussian_rasterizer](https://github.com/VladimirYugay/gaussian_rasterizer) from "Gaussian-SLAM", this fork adds:

* Uncertainty map rendering via per-splat appearance variance.
* Depth backpropagation and alpha rendering.

<!-- ## Install

```bash
pip install .
``` -->

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tran2026varsplat,
  title   = {VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM},
  author  = {Tran, Anh Thuan and Kosecka, Jana},
  booktitle = {CVPR},
  year    = {2026}
}
```

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
