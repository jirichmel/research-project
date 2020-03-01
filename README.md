## Repository Description
The following contents serve as description of the data and scripts found in this repository. This repository contains datasets in .csv form and relevant .ipynb python scripts.
(Note: The scripts are .ipynb and github does not render such files. To render these files on github in your browser use the Firefox/Chromium/Chrome extension: https://github.com/iArunava/NoteBook-Buddy/)

All the datasets in this repository were created using the datafiles provided by Dr. Luca Ghiringhelli of the Theory Department of Fritz Haber Institute of the Max Planck Society in Berlin and the datafiles from a Kaggle competition available at: https://www.kaggle.com/c/nomad2018-predict-transparent-conductors/data

**The author would like to thank Dr. Luca Ghiringhelli for providing the datafiles.**

## .csv Files
The datasets in the repository are structured into two main groups: train data and test data. There are 2400 train materials and 600 test materials. The split of the data is the same as in the Kaggle dataset for convinience and easy comparison of results.

These datasets contain not only the data after the calculation but also the atomic positions, lattice vectors and energies during the converge of the atomic positions (relaxation). This additional data was not included in the Kaggle dataset.

There are two folders: train_relaxation_csv and test_relaxation_csv.

The numbering in the files works as follows: 
* **id** is the number of the material (the same numbering as in the Kaggle dataset).
* **relaxation_step_number** labels the given relaxation step of the calculation of the particular material. Therefore, each material is characterized by one **id** number and a few **relaxation_step_number**s whose amount differs (Note: Some materials do not have any Relaxation steps and therefore do not have any relaxation step numbers. They are therefore absent in the four datasets which can be found in the folders for both train and test). The values belonging to the last relaxation step number are the final values of the calculation.

id | relaxation_step_number
------------ | -------------
2 | 1
2 | 2

Each folder contains 4 .csv files:
* **Item atoms_frac_xyz_relaxation_DATA.csv** - contains the fractional coordinates (fraction of given the lattice vectors) during the relaxation.

* **atoms_xyz_relaxation_DATA.csv** - contains the atomic positions of the atoms.

* **energy_relaxation_DATA.csv** - contains the last HOMO-LUMO value of the particular relaxation step and the last formation energy value which is calculated using the last Total Energy value of the particluear relaxation step.

* **lattice_vector_relaxation.csv** - contains the lattice vector values during the relaxation.

In addition to these two folders, two .csv files called **general_train_DATA.csv** and **general_test_DATA.csv** are included. They contain the spacegroup data, number of atoms and the cation percetages calculated using:

<img src="https://render.githubusercontent.com/render/math?math=x = \frac{ n_{Al} }{ n_{Al} %2B n_{Ga} %2B n_{In} } ">
<img src="https://render.githubusercontent.com/render/math?math=y = \frac{ n_{Ga} }{ n_{Al} %2B n_{Ga} %2B n_{In} } ">
<img src="https://render.githubusercontent.com/render/math?math=z = \frac{ n_{In} }{ n_{Al} %2B n_{Ga} %2B n_{In} } ">

The values lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree from the Kaggle dataset were removed in this dataset.

## kaggle_data_handling.ipynb Script
The script used for the extraction the data in this dataset from the provided datafiles. The script uses a folder structure of the provided datafiles. Modification of the paths to the files is needed if the script is to be used.
