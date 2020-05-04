#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

def get_forces_tensors_and_pressure(raw_data_file, ajdi, df_frac):
    with open(raw_data_file, 'r') as file:
        dic_1={}
        dic_2={}
        dummy=[]
        rsn_1 = 0
        rsn_2 = 0
        rsn_1_lst=[]
        rsn_2_lst=[]
        ajdi_1_lst=[]
        ajdi_2_lst=[]
        x=[]
        y=[]
        z=[]
        dic={}
        xx=[]
        xy=[]
        xz=[]
        yy=[]
        yz=[]
        zz=[]
        pressure=[]
        while True:
            try:
                line = next(file)
                if "Analytical stress tensor - Symmetrized" in line:
                    [next(file) for _ in range(4)]
                    dummy=[]
                    for _ in range(3):
                        line = next(file)
                        dummy.append(line.split()[2:-1])
                    ajdi_1_lst.append(ajdi)
                    rsn_1_lst.append(rsn_1)
                    xx.append(dummy[0][0])
                    xy.append(dummy[0][1])
                    xz.append(dummy[0][2])
                    yy.append(dummy[1][1])
                    yz.append(dummy[1][2])
                    zz.append(dummy[2][2])
                    next(file)
                    line = next(file)
                    pressure.append(line.split()[2:-2][0])
                    rsn_1 += 1

                if "Total atomic forces" in line:
                    line = next(file)
                    #while "  |" in line:
                    for _ in range(len(df_frac[(df_frac["id"]==ajdi) & (df_frac["relaxation_step_number"]==rsn_2)])):
                        data = line.split()[2:]
                        ajdi_2_lst.append(ajdi)
                        rsn_2_lst.append(rsn_2)
                        x.append(data[0])
                        y.append(data[1])
                        z.append(data[2])
                        line = next(file)
                    rsn_2 += 1

            except StopIteration:
                dic = {"id": df_frac.id.loc[(df_frac["id"]==ajdi)].to_list(), "relaxation_step_number": df_frac.relaxation_step_number.loc[(df_frac["id"]==ajdi)].to_list(), "species": df_frac.species.loc[(df_frac["id"]==ajdi)].to_list(), "Fx": x, "Fy": y, "Fz": z}
                df_forces = pd.concat([pd.Series(v, name=k) for k, v in dic.items()], axis=1)

                dic = {"id": ajdi_1_lst, "relaxation_step_number": rsn_1_lst, "xx": xx, "yy": yy, "zz": zz, "xy": xy, "xz": xz, "yz": yz, "pressure": pressure}
                df_stress_tensor_and_pressure = pd.concat([pd.Series(v, name=k) for k, v in dic.items()], axis=1)
                print("All tensors, pressures and forces from id", ajdi, "extracted.", rsn_1, rsn_2)
                return df_stress_tensor_and_pressure, df_forces


# test set

df_stress_tensor_and_pressure = pd.DataFrame(columns=['id', 'relaxation_step_number', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'pressure'])
df_forces = pd.DataFrame(columns=['id', 'relaxation_step_number', 'species', 'Fx', 'Fy', 'Fz'])

df_frac = pd.read_csv("/home/jurka/research-project/test/relaxation/with_all_zeros/atoms_frac_xyz_with_zeros.csv")

list_of_file_numbers = [i for i in range(1,601)]

for i in list_of_file_numbers:
    path = "".join(["/home/jurka/Desktop/REE/nomad_kaggle_bandfiles/test_bandfiles/just_bandfiles/", str(i), "/band.out"])
    
    df1, df2 = get_forces_tensors_and_pressure(path, i, df_frac)
     
    df_stress_tensor_and_pressure = pd.concat([df_stress_tensor_and_pressure, df1]).reset_index(drop=True)
    df_forces = pd.concat([df_forces, df2]).reset_index(drop=True)
        
df_stress_tensor_and_pressure.to_csv("/home/jurka/research-project/test/additional/stress_tensor_and_pressure.csv", index = False)
df_forces.to_csv("/home/jurka/research-project/test/additional/forces.csv", index = False)


# train set

df_stress_tensor_and_pressure = pd.DataFrame(columns=['id', 'relaxation_step_number', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'pressure'])
df_forces = pd.DataFrame(columns=['id', 'relaxation_step_number', 'species', 'Fx', 'Fy', 'Fz'])

df_frac = pd.read_csv("/home/jurka/research-project/train/relaxation/with_all_zeros/atoms_frac_xyz_with_zeros.csv")

list_of_file_numbers = [i for i in range(1,2401)]   

# Problematic datafiles:
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(464)])
del list_of_file_numbers[list_of_file_numbers.index(464)] # 464. is problematic and will not be included in the dataset
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(2189)])
del list_of_file_numbers[list_of_file_numbers.index(2189)] # 2189. is problematic and will not be included in the dataset

# Removing duplicates from the train set:
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(126)])
del list_of_file_numbers[list_of_file_numbers.index(126)] # id 395/126 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(1215)])
del list_of_file_numbers[list_of_file_numbers.index(1215)] # id 1215/1886 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(353)])
del list_of_file_numbers[list_of_file_numbers.index(353)] # id 2075/353 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(308)])
del list_of_file_numbers[list_of_file_numbers.index(308)] # id 308/2154 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(531)])
del list_of_file_numbers[list_of_file_numbers.index(531)] # id 531/1379 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(2319)])
del list_of_file_numbers[list_of_file_numbers.index(2319)] # id 2319/2337 is duplicate. One removed.
print("Removing material id",list_of_file_numbers[list_of_file_numbers.index(2370)])
del list_of_file_numbers[list_of_file_numbers.index(2370)] # id 2370/2333 is duplicate. One removed.

for i in list_of_file_numbers:
    path = "".join(["/home/jurka/Desktop/REE/nomad_kaggle_bandfiles/train_bandfiles/just_bandfiles/", str(i), "/band.out"])
    
    df1, df2 = get_forces_tensors_and_pressure(path, i, df_frac)
     
    df_stress_tensor_and_pressure = pd.concat([df_stress_tensor_and_pressure, df1]).reset_index(drop=True)
    df_forces = pd.concat([df_forces, df2]).reset_index(drop=True)
        
df_stress_tensor_and_pressure.to_csv("/home/jurka/research-project/train/additional/stress_tensor_and_pressure.csv", index = False)
df_forces.to_csv("/home/jurka/research-project/train/additional/forces.csv", index = False)

