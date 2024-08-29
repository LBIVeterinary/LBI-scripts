# -*- coding: utf-8 -*-
import pandas as pd
import os, sys, random

_input = "Probiotics.csv"
parcial = "Probiotics_filtred_1.csv"
_output = "Probiotics_filtred_2.csv"

df = pd.read_csv(_input, sep=",")

remove = []
for i in df.index:
    if (type(df["GenBank FTP"][i]) == float) and (type(df["RefSeq FTP"][i]) == float):
        remove.append(i)
df.drop(df.index[remove], inplace=True)

duplicated = df.index.groupby(df['Strain'])
remove = []
for strain in duplicated:
    inst = list(duplicated[strain])
    if len(inst) > 1:
        organisms = []
        for index in inst:
            genus = df["#Organism Name"][index].split(" ")[0]
            epiteto = df["#Organism Name"][index].split(" ")[1]
            specie = f"{genus} {epiteto}"
            organisms.append(specie)
        for posicao in range(1, len(organisms)):
            if organisms[posicao] == organisms[posicao-1]:
                if inst[posicao] not in remove:
                    remove.append(inst[posicao])
execao = []
for i in remove:
    try:
        df.drop(df.index[i], inplace=True)
    except:
        execao.append(i)
                
df.to_csv(parcial, sep = ",", header=True, index=False, quoting=1)

file = open(parcial, "rt")
lines = file.readlines()
file.close()

index = random.sample(lines[1:], 3000)

out = open(_output, "w")
out.write(lines[0])
out.close()

out = open(_output, "a")
ind = 0
for i in index:
	out.write(i)

out.close()

teste = pd.read_csv(_output, sep=",")
print(list(teste["#Organism Name"]).count("Lactococcus petauri"))
