UncurlNet: Deep matrix factorization for single-cell RNA-seq
============================================================

Installation
------------

requirements: pytorch, uncurl

Usage
-----

How it works:


Ideas
-----

How to do batch correction:
- use separate w-encoder for each batch, but hold M constant?

- a problem comes in the objective - how do we allow the reconstruction to be different from the original data in a systematic way? Maybe instead of using a separate w-encoder, we could have an additional post-encoder layer that encodes the batch effect. The batch effect is essentially an affine transform of the data?

- dealing with cell-specific effects such as library size: could we just have a multiplicative layer? Like, a layer that, given the cell, returns a constant multiplicative weight that's multiplied to the end?
