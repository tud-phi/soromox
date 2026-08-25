# Citing SoRoMoX

## Primary Paper

If you use SoRoMoX in academic work, please cite the associated preprint:

```bibtex
@misc{stolzle2026soromox,
  title = {{SoRoMoX}: Fast, Differentiable, and Parallelizable Soft Robot Models},
  author = {Maximilian St{\"o}lzle and Solange Gribonval and Daniel {Feliu-Talegon} and Vito Daniele Perfetta and Michele Martini and Chuhan Zhang and Kiwan Wong and Mohammed Tarnini and Anup Teejo Mathew and Federico Renda and Daniela Rus and Cosimo {Della Santina}},
  year = {2026},
  eprint = {2608.06650},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  doi = {10.48550/arXiv.2608.06650},
  url = {https://arxiv.org/abs/2608.06650},
}
```

The latest preprint is available at
[arXiv:2608.06650](https://arxiv.org/abs/2608.06650).

## Reproducibility and Software Version

When SoRoMoX generates results reported in a publication, also state the exact
package version. It is available at runtime as:

```python
import soromox

print(soromox.__version__)
```

If reproducibility depends materially on a particular release, we encourage an
additional software citation for that release:

```bibtex
@software{soromox_v0_3_0,
  title = {{SoRoMoX}: Soft Robot Models in {JAX}},
  author = {Maximilian St{\"o}lzle and Solange Gribonval and Daniel {Feliu-Talegon} and Vito Daniele Perfetta and Michele Martini and Chuhan Zhang and Kiwan Wong and Daniela Rus and Cosimo {Della Santina}},
  year = {2026},
  version = {0.3.0},
  url = {https://github.com/tud-phi/soromox/releases/tag/v0.3.0},
}
```

Use the entry associated with the version that produced the results rather than
substituting the newest release.

## Additional Model and Method Citations

These references are conditional: cite them when the corresponding model or
controller implementation is material to the work.

### Planar HSA Model

For the kinematic and dynamic model of the planar Handed Shearing Auxetics
robot, also cite:

```bibtex
@inproceedings{stolzle2024experimental,
  title = {An Experimental Study of Model-Based Control for Planar Handed Shearing Auxetics Robots},
  author = {Maximilian St{\"o}lzle and Daniela Rus and Cosimo {Della Santina}},
  booktitle = {Experimental Robotics: The 18th International Symposium},
  series = {Springer Proceedings in Advanced Robotics},
  volume = {30},
  pages = {153--167},
  publisher = {Springer},
  year = {2024},
  doi = {10.1007/978-3-031-63596-0_14},
}
```

### Model-Based Controllers

If the controller formulations or implementation details described in Chapter 2
of the following thesis are material to the work, also cite:

```bibtex
@phdthesis{stolzle2025phdthesis,
  title = {Safe yet Precise Soft Robots: Incorporating Physics into Learned Models for Control},
  author = {Maximilian St{\"o}lzle},
  year = {2025},
  type = {Dissertation},
  school = {Delft University of Technology},
  doi = {10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da},
  isbn = {978-94-6384-836-7},
}
```

## Citation and License

The citation guidance above is a scholarly request. SoRoMoX is distributed
under the MIT License; redistribution requirements are defined by
[`LICENSE.txt`](https://github.com/tud-phi/soromox/blob/main/LICENSE.txt).
