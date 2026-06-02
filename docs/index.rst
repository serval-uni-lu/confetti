.. image:: _static/confetti-logo.svg
   :alt: CONFETTI logo
   :align: center
   :class: only-light

.. image:: _static/confetti-white.svg
   :alt: CONFETTI logo
   :align: center
   :class: only-dark

Counterfactual Explanations for Time Series and Tabular Data
====================================================================


**CONFETTI** is a multi-objective method for generating **counterfactual explanations** for
**multivariate time series** and **tabular data** classifiers.
It identifies the most influential features or temporal regions, constructs a minimal perturbation
using the nearest unlike neighbour (NUN), and optimizes it under multiple objectives to produce
explanations that are **sparse**, **realistic**, and **confidence-increasing**.

The method is model-agnostic and works with any **Keras**, **PyTorch**, or **scikit-learn** classifier.

Installation
------------

To install the PyPI release:

.. code-block:: bash

    pip install confetti-ts

Features
--------
- 🐍 Compatible with Python 3.12+
- 🎯 Multi-objective counterfactual generation using NSGA-III
- 📊 **Time series**: works with any Keras or scikit-learn multivariate time series classifier
- 📋 **Tabular data**: works with any classifier exposing ``predict_proba`` or ``predict``
- 🔗 **Relation constraints on tabular data**: enforce domain rules on counterfactuals (e.g. ``age <= retirement_age``)
- 🔥 Optional use of CAMs for feature-weighted perturbations (time series)
- ⚡ Rust-accelerated backend for distances, NSGA-III, and constraint evaluation
- 🧪 Generates multiple diverse counterfactuals per instance
- ⚙️ Parallelized counterfactual generation
- 🧰 Built-in utilities for:

  - 📄 loading and preparing time series datasets
  - 🔍 extracting CAM feature weights
  - 📊 visualizing generated explanations

License
-------

CONFETTI is released under the terms of the MIT License.

.. toctree::
   :hidden:
   :maxdepth: 2

   usage
   example
   tabular_usage
   tabular_example
   constraints
   api/index

Citing CONFETTI
-----------------
If you use CONFETTI in your research, please cite the following paper:

.. code-block:: bibtex

    @inproceedings{cetina2026counterfactual,
      title={Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification},
      author={Cetina, Alan Gabriel Paredes and Benguessoum, Kaouther and Lourenco, Raoni and Kubler, Sylvain},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={40},
      number={21},
      pages={17393--17400},
      year={2026}
    }
