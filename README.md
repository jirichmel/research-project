## Repository Description
The following contents serve as description of the data and scripts found in this repository. This repository contains datasets in .csv form and relevant python scripts in .ipynb and .py format.
(Note: The scripts are .ipynb and github does not render such files. To render these files on github in your browser use the Firefox/Chromium/Chrome extension: https://github.com/iArunava/NoteBook-Buddy/)

All the datasets in this repository were created using the datafiles provided by Dr. Luca Ghiringhelli of the Theory Department of Fritz Haber Institute of the Max Planck Society in Berlin and the datafiles from a Kaggle competition available at: https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data

**The author would like to thank Dr. Luca Ghiringhelli for providing the datafiles.**

## The Folder Structure Explained
The datasets in the repository are structured into two main groups: train data and test data. Originally, there were 2400 train materials and 600 test materials. There are 7 duplicates and 2 datafiles were corrupted which makes it 2391 train materials (for more details view the end of this description where the section **Known Issues** is located) and 600 test materials. The split of the data is the same as in the Kaggle dataset for convinience and easy comparison of results.

These datasets contain not only the data after the calculation but also the atomic positions, lattice vectors and energies during the converge of the atomic positions (relaxation). This additional data was not included in the Kaggle dataset.

There are two folders: **train** and **test**.

The numbering in the .csv files works as follows: 
* **id** is the number of the material (the same numbering as in the Kaggle dataset).
* **relaxation_step_number** labels the given relaxation step of the calculation of the particular material. Therefore, each material is characterized by one **id** number and a few **relaxation_step_number**s whose amount differs (Note: Some materials do not have any Relaxation steps and therefore do not have any relaxation step numbers. Therefore, they would be absent in the relaxation datasets. In such situation, the final atomic coordinates and energies are included with **relaxation_step_number** equal 0 to distiguish the nature of this datapoint in some way). The values belonging to the last relaxation step number are the final values of the calculation.
An example of the start of the head of the .csv file with relaxation data:

id | relaxation_step_number | ...
------------ | ------------- | -------------
2 | 1 | ...
2 | 2 | ...
. | . | ...
. | . | ...
. | . | ...

The **test** and **train** folders contain 3 folders each:
* **directory_tree** - contains a directory tree, where the parent directory of every material is named after its **id** and the coresponding final data files are in a child directory of the same name and the relaxation datafiles are in a child directory which has the name of the form **id.relaxation_step_number**. E.g. in **train**, the third material would have a parent folder named **3** its final values in a child folder named **3** and its first relaxation data in a different child folder named **3.1**.
* **final** - contains 4 .csv files of the final values: **atoms_frac_xyz.csv, atoms_xyz.csv, energy.csv, lattice_vector.csv**
* **relaxation** - contais 5 .csv files of the relaxation values plus a file with general data describing the material: **atoms_frac_xyz_relaxation.csv, energy_relaxation.csv, lattice_vector_relaxation.csv, atoms_xyz_relaxation.csv, general.csv**

Each **final** folder contains 4 .csv files:
* **atoms_frac_xyz.csv** - contains the final fractional coordinates (fraction of given the lattice vectors).

* **atoms_xyz.csv** - contains the final atomic positions of the atoms.

* **energy.csv** - contains the final HOMO-LUMO value and the final formation energy value which is calculated using the final Total Energy value.

* **lattice_vector.csv** - contains the final lattice vector values.


Each **relaxation** folder contains 5 .csv files:
* **atoms_frac_xyz_relaxation.csv** - contains the fractional coordinates (fraction of given the lattice vectors) during the relaxation.

* **atoms_xyz_relaxation.csv** - contains the atomic positions of the atoms.

* **energy_relaxation.csv** - contains the last HOMO-LUMO value of the particular relaxation step and the last formation energy value which is calculated using the last Total Energy value of the particluear relaxation step.

* **lattice_vector_relaxation.csv** - contains the lattice vector values during the relaxation.

* **general.csv** - contains the spacegroup data, number of atoms and the cation percetages calculated using:

<img src="https://render.githubusercontent.com/render/math?math=x = \frac{ n_{Al} }{ n_{Al} %2B n_{Ga} %2B n_{In} } ">
<img src="https://render.githubusercontent.com/render/math?math=y = \frac{ n_{Ga} }{ n_{Al} %2B n_{Ga} %2B n_{In} } ">
<img src="https://render.githubusercontent.com/render/math?math=z = \frac{ n_{In} }{ n_{Al} %2B n_{Ga} %2B n_{In} } ">

The values lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree from the Kaggle dataset were removed in this dataset.

## kaggle_data_handling Script
The script used for the extraction the data in this dataset from the provided datafiles. The script uses the folder structure of the provided datafiles. Modification of the paths to the files is needed if the script is to be reused. The scripts uses parallelization to multiple CPUs using the ray framework. The script is not optimized for maximum efficiency. Parallel computation is recommended. The OLS_oxide_energies script contains a method to do an OLS fit of the 3 total energies needed to calculate the formation energy.
## Known Issues
* In **train**, the datafile number (**id**) 464 seems to be incomplete or corrupted. The datapoint was excluded from the datasets.
* In **train**, the datafile number (**id**) 2189 seems to be incomplete or corrupted. The datapoint was excluded from the datasets.

Source of the claims listed below: https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/discussion/47998
* In **train**, the materials with **id** 395 and 126 are duplicate. 126 was removed.
* In **train**, the materials with **id** 1215 and 1886 are duplicate. 1215 was removed.
* In **train**, the materials with **id** 2075 and 353 are duplicate. 353 was removed.
* In **train**, the materials with **id** 308 and 2154 are duplicate. 308 was removed.
* In **train**, the materials with **id** 531 and 1379 are duplicate. 531 was removed.
* In **train**, the materials with **id** 2319 and 2337 are duplicate. 2319 was removed.
* In **train**, the materials with **id** 2370 and 2333 are duplicate. 2370 was removed.

## Matlab files:
Exp_01.m: Least squares (LS) for 3descriptors (ratios of Al, Ga, In among metals), test and train sets as pre-defined, RMSE: 0.0931
Exp_02.m: LS with Unigrams (Al, In, Ga are of type 2-8), RMSE: 0.0767
Exp_04.m: LS with Unigrams for Al, In, Ga, O, RMSE: 0.0714
ReadData.m: Input of data, save the workspace...
Exp_10.m, 10a.m, 10b.m... 11b.m: sum over unigrams of Al, In, Ga of 1, r, r^2 and unigrams of O, RMSE: 0.445

## To Do
Decide whether to include the following dataset and if so, how:

https://repository.nomad-coe.eu/app/gui/dataset/id/nn56nNWzSOGmuf0ptRMymg
https://repository.nomad-coe.eu/app/gui/dataset/id/K1QTEYCNRkaeb7JQzgihsw

