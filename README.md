# AlphaFold2-RAVE v 1.0

## Demonstrative colab notebook can be found here -> [Complete run through on CSP system](https://colab.research.google.com/github/bodhivani/temprep/blob/main/fullrunthrough_CSP.ipynb)
This notebook aims to show how efficient our method is in predicting an effective RC (SPIB) from the amino acid sequence of a protein of interest(CSP).

We provide a `light_version` option in our notebook to get a brief overview and the essence of our protocol. 

| light_version  |   Time   | MD simulation  | SPIB |
|----------------|:--------:|----------------|------|
|      True      |  1  min  |       :x:      |   ✅ |
|      False     |  ~ 3 hrs |        ✅      |   ✅ |


## Motivation
This protocol is essentially a methodology that combines two schools of thought: structure prediction, and enhanced sampling to preserve thermodynamics. In this repository, we demonstrate one instance of such a methodology. For structure prediction, we use AlphaFold2, or more specifically ColabFold. We introduce stochasticity to the ColabFold algorithm by decreasing the MSA cluster size and generating structures with multiple random seeds<sup>\*</sup>. For thermodynamics, we perform molecular dynamics simulations with metadynamics. To bias the simulations, we learn collective variables from the now-diverse set of structures using SPIB<sup>\*</sup>.

###### <sup>*</sup> The changes to the ColabFold algorithm, as well as the additional code required to use the SPIB for biasing have been provided in this github and their use is demonstrated in the colab files

## Components

* `ravefuncs.py` - functions required to run this instance of our methodology
* `sample_config.ini` - basic input configurations required for SPIB
* [`fullrunthrough_CSP.ipynb`](https://colab.research.google.com/github/bodhivani/temprep/blob/main/fullrunthrough_CSP.ipynb) - our full protocol demonstrated to run on the colab GPU server to show improvement in sampling in a short time frame compared to unbiased simulation. This may take 2-4h to run.
* `litedemo.ipynb` - a tutorial of our protocol without running simulations, explaining how to build required code and files to adapt the methodology to your system and computational resources
* `CSP_data` - files required to run parts of fullrunthrough_CSP.ipynb and litedemo.ipynb on the benchmark system cold-shock protein 1HZB.

## Citation

Please cite the following reference if using this protocol with or without the provided code:

* "AlphaFold2-RAVE: From sequence to Boltzmann ensemble"
Bodhi P. Vani, Akashnathan Aranganathan, Dedi Wang, Pratyush Tiwary
bioRxiv 2022.05.25.493365; doi: https://doi.org/10.1101/2022.05.25.493365

* "State predictive information bottleneck", Dedi Wang and Pratyush Tiwary, J. Chem. Phys. 154, 134111 (2021) https://doi.org/10.1063/5.0038198

If the provided code is used for the protocol, please also cite:

* M. Mirdita, K. Schütze, Y. Moriwaki, L. Heo, S. Ovchinnikov, and M. Steinegger, “ColabFold: making protein folding accessible to all,” Nature Methods, vol. 19, no. 6. Springer Science and Business Media LLC, pp. 679–682, May 30, 2022. doi: 10.1038/s41592-022-01488-1.
* P. Eastman et al., “OpenMM 7: Rapid development of high performance algorithms for molecular dynamics,” PLOS Computational Biology, vol. 13, no. 7. Public Library of Science (PLoS), p. e1005659, Jul. 26, 2017. doi: 10.1371/journal.pcbi.1005659.
* “Promoting transparency and reproducibility in enhanced molecular simulations,” Nature Methods, vol. 16, no. 8. Springer Science and Business Media LLC, pp. 670–673, Jul. 30, 2019. doi: 10.1038/s41592-019-0506-8.
* A. Barducci, G. Bussi, and M. Parrinello, “Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method,” Physical Review Letters, vol. 100, no. 2. American Physical Society (APS), Jan. 18, 2008. doi: 10.1103/physrevlett.100.020603.


Note: This will be updated on peer reviewed acceptance and parameters chosen for robustness
Note: Toolset will be uploaded soon

## License

MIT License

Copyright (c) 2023 Tiwary Research Group

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
