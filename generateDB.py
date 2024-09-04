#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from datetime import datetime as dt
import string, os, argparse, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Bio import SeqIO
import subprocess as sbp
import shutil as sh
import shap

def arguments():
	parser = argparse.ArgumentParser(description='Process some files.')
	parser.add_argument('--matrix', type=str, required=True, help='Path to the matrix file')
	parser.add_argument('--outdir', type=str, required=True, help='Output directory')
	parser.add_argument('--ffn', type=str, required=True, help='Path to the ffn file')
	parser.add_argument('--faa', type=str, required=True, help='Path to the faa file')
	parser.add_argument('--iterations', type=int, required=True, help='Number of iterations for SVM and Random Forest')
	parser.add_argument('--percentage', type=float, required=True, help='Minimum percentage for consider a gene for each set in SVM consensus prediction')
	parser.add_argument('--dataset_per', type=float, required=True, help='Minimum percentage of present genes within each dataset (positive and negative). It determines the lenght of the final datasets')
	parser.add_argument('--kegg', type=str, required=True, help='Path to the kegg dictionary')
	parser.add_argument('--cog', type=str, required=True, help='Path to the cog dictionary')
	parser.add_argument('--force', '-f', type=bool, nargs='?', const=True, default=False, help='Force output directory overwrite')

	parser.add_argument('--lr', type=bool, nargs='?', const=True, default=False, help='Run Logistic Regression Analysis')
	parser.add_argument('--svm', type=bool, nargs='?', const=True, default=False, help='Run Support Vector Machine Analysis')
	parser.add_argument('--rf', type=bool, nargs='?', const=True, default=False, help='Run Random Forest Analysis')	

	args = parser.parse_args()

	files = {
	    'matrix': args.matrix,
	    'ffn': args.ffn,
	    'faa': args.faa,
	    'iterations': args.iterations,
	    'percentage': args.percentage,
	    'dataset_per': args.dataset_per,
	    'kegg': args.kegg,
	    'cog': args.cog,
	    'svm': args.svm,
	    'lr': args.lr,
	    'rf': args.rf
	}

	for name, file_path in files.items():
		if type(file_path) != str:
			continue
		if os.path.exists(file_path) == False:
			raise FileNotFoundError(f"Error: {name} file does not exist at {file_path}")

	if args.force == True:
		files['outdir'] = args.outdir
	else:
		if os.path.isdir(args.outdir) == False:
			files['outdir'] = args.outdir
		else:
			raise Exception(f"Error: directory does exist at {args.outdir}. Use --force or -f.")

	generate_outdir(args.outdir)

	write_log(files ," ".join(list(sys.argv)))

	write_log(files, f'Matrix file: {args.matrix}')
	write_log(files, f'FFN file: {args.ffn}')
	write_log(files, f'FAA file: {args.faa}')
	write_log(files, f'Iterations: {args.iterations}')
	write_log(files, f"SVM Min Percentage: {args.percentage}")
	write_log(files, f'Minimum Percentage: {args.dataset_per}')
	write_log(files, f'KEGG Dictionary: {args.kegg}')
	write_log(files, f'COG Dictionary: {args.cog}')
	write_log(files, f'Output directory: {args.outdir}')
	return files

def generate_outdir(inp):
	if os.path.isdir(inp):
		if (inp == '.') or (inp == './') or (inp == '..') or (inp == '../'):
			raise Exception(f"Error: it's not possible to delete {inp}")
		else:
			sh.rmtree(inp)
			os.makedirs(inp)
			os.makedirs(f'{inp}{os.sep}iterations')
	else:
		os.makedirs(inp)
		os.makedirs(f'{inp}{os.sep}iterations')

def extract_fasta(alfas_df, seqs, extension):
	prefix, extensao = os.path.splitext(alfas_df)
	df = pd.read_csv(alfas_df, sep='\t', index_col="locus_tag")
	output_file = open(f"{prefix}.{extension}", "w")
	for locus_tag in df.index:
			if locus_tag in seqs:
					SeqIO.write(seqs[locus_tag], output_file, "fasta")
	output_file.close()

def plot_importance(alfas, title, minimum, maximum, y_axis, out):
	plt.clf()
	plt.figure(figsize=(15, 10))
	plt.plot(np.arange(1, len(alfas) + 1), alfas, marker='o', linestyle='', color='#4B0082')
	plt.xlabel('Locus Index', fontsize=20)
	plt.ylabel(y_axis, fontsize=20)
	plt.title(title, fontsize=24)
	plt.grid(True)
	plt.axhline(y=minimum, color='#16ff16', linestyle='--')
	plt.axhline(y=maximum, color='red', linestyle='--')
	plt.savefig(f'{out}_importance_plot.png', dpi=300)

def write_log(inp, phrase):
	out = f'{inp["outdir"]}{os.sep}log.log'
	now = dt.now().strftime("%d-%m-%Y %H-%M-%S")
	print(f'{now}\t{phrase}')
	with open(out, 'a') as out_file:
		if phrase.endswith('\n') == False:
			phrase = f'{phrase}\n'
		out_file.write(f'{now}\t{phrase}')
		out_file.close()

def run_lr(inp, out):
	out = f'{inp["outdir"]}{os.sep}{out}'
	write_log(inputs, 'Running Logistic Regression.')
	write_log(inputs, 'Reading original matrix.')
	df = pd.read_csv(inp["matrix"], sep='\t', index_col="Gene").transpose()
	y = [0 if i.startswith("0_") else 1 for i in df.index]
	write_log(inputs, 'Fitting Logistic Regression model.')
	model = LogisticRegression(max_iter=300)
	model.fit(df, y)
	alfa = model.coef_
	alfa_values = alfa.flatten()
	df_alfa = pd.DataFrame(alfa, columns=df.columns).transpose()
	df_alfa.index.name="locus_tag"
	df_alfa.rename(columns={0: 'alphas'}, inplace=True)
	df_alfa = df_alfa.sort_values(by=['alphas'], ascending=False, inplace=False)
	df_alfa.to_csv(f'{out}_raw.tsv', sep='\t', index=True)
	expected_amout = len(df.columns) * inp["dataset_per"] / 100
	write_log(inputs, f"Number of genes expected: {expected_amout}")
	min_alfa = df_alfa['alphas'][df_alfa.index[int(expected_amout)]]
	df_alfas_above = df_alfa[df_alfa['alphas'] >= min_alfa]
	df_alfas_above.to_csv(f'{out}_positive.tsv', sep='\t', index=True)
	df_alfa = df_alfa.sort_values(by=['alphas'], ascending=True, inplace=False)
	max_alfa = df_alfa['alphas'][df_alfa.index[int(expected_amout)]]
	df_alfas_below = df_alfa[df_alfa['alphas'] <= max_alfa]
	df_alfas_below.to_csv(f'{out}_negative.tsv', sep='\t', index=True)
	write_log(inputs, 'Reading proteins fasta.')
	sequences = SeqIO.to_dict(SeqIO.parse(inp['faa'], "fasta"))
	extract_fasta(f'{out}_positive.tsv', sequences, "faa")
	extract_fasta(f'{out}_negative.tsv', sequences, "faa")
	del sequences
	write_log(inputs, 'Reading genes fasta.')
	sequences = SeqIO.to_dict(SeqIO.parse(inp['ffn'], "fasta"))
	extract_fasta(f'{out}_positive.tsv', sequences, "ffn")
	extract_fasta(f'{out}_negative.tsv', sequences, "ffn")
	del sequences
	plot_importance(alfa_values,
		'Alpha Values Obtained by Logistic Regression',
		min_alfa,
		max_alfa,
		'Coefficient values',
		out)

def run_svm(inp, out):
	out = f'{inp["outdir"]}{os.sep}{out}'
	out_ite = f'{inp["outdir"]}{os.sep}iterations{os.sep}'
	write_log(inputs, 'Running SVM.')
	df = pd.read_csv(inp["matrix"], sep='\t', index_col="Gene").transpose()
	y = [0 if i.startswith("0_") else 1 for i in df.index]
	iterations = {column: 0 for column in df.columns}
	for ind in range(0, inp["iterations"]):
		write_log(inputs, f'Running SVM iteration {ind}')
		model = SGDClassifier(loss='hinge')
		scaler = StandardScaler()
		X = df.values
		X = scaler.fit_transform(X)
		model.partial_fit(X, y, classes=np.array([0, 1]))
		coefficients = model.coef_.flatten()
		attribute_names = df.columns
		importance_df = pd.DataFrame({
			'locus_tag': attribute_names,
			'Coefficient': coefficients})
		importance_df.set_index('locus_tag', inplace=True)
		importance_df = importance_df.sort_values(by=['Coefficient'], ascending=False, inplace=False)
		expected_amout = len(df.columns) * inp["dataset_per"] / 100
		min_coef = importance_df['Coefficient'][importance_df.index[int(expected_amout)]]
		df_coefficients_above = importance_df[importance_df["Coefficient"] >= min_coef]
		df_coefficients_above.to_csv(f'{out_ite}svm_positive_{ind}.tsv', sep="\t", index=True)
		for locus in df_coefficients_above.index:
			iterations[locus] = iterations[locus] + 1
		importance_df = importance_df.sort_values(by=['Coefficient'], ascending=True, inplace=False)
		max_coef = importance_df['Coefficient'][importance_df.index[int(expected_amout)]]
		df_coefficients_below = importance_df[importance_df["Coefficient"] <= max_coef]
		df_coefficients_below.to_csv(f'{out_ite}svm_negative_{ind}.tsv', sep="\t", index=True)
		for locus in df_coefficients_below.index:
                        iterations[locus] = iterations[locus] - 1
	df_iterations = pd.DataFrame(list(iterations.items()), columns=['locus_tag', 'occurrences'])
	df_iterations.to_csv(f"{out}_occurrences_raw.tsv", sep="\t", index=False)
	with open(f'{out}_positive.tsv', 'w') as output_file:
		output_file.write('locus_tag\toccurences\n')
		for locus in iterations:
			if iterations[locus] >= inp["iterations"]*inp["percentage"]/100:
				output_file.write(f'{locus}\t{iterations[locus]}\n')
	with open(f'{out}_negative.tsv', 'w') as output_file:
		output_file.write('locus_tag\toccurences\n')
		for locus in iterations:
			if iterations[locus] <= (0 - (inp["iterations"]*inp["percentage"]/100)):
				output_file.write(f'{locus}\t{iterations[locus]}\n')
	write_log(inputs, 'Reading proteins fasta.')
	sequences = SeqIO.to_dict(SeqIO.parse(inp['faa'], "fasta"))
	extract_fasta(f'{out}_positive.tsv', sequences, "faa")
	extract_fasta(f'{out}_negative.tsv', sequences, "faa")
	del sequences
	write_log(inputs, 'Reading genes fasta.')
	sequences = SeqIO.to_dict(SeqIO.parse(inp['ffn'], "fasta"))
	extract_fasta(f'{out}_positive.tsv', sequences, "ffn")
	extract_fasta(f'{out}_negative.tsv', sequences, "ffn")
	del sequences
	plot_importance(list(iterations.values()),
				'Occurrences Obtained by Support Vector Machine',
				inp["iterations"]*inp["percentage"]/100,
				0-(inp["iterations"]*inp["percentage"]/100),
				'Number of occurrences',
				out)

def run_rf(inp, out):
	out = f'{inp["outdir"]}{os.sep}{out}'
	out_ite = f'{inp["outdir"]}{os.sep}iterations{os.sep}'
	write_log(inputs, 'Running Random Forest.')
	write_log(inputs, 'Reading original matrix.')
	df = pd.read_csv(inp["matrix"], sep='\t', index_col="Gene").transpose()
	y = [0 if i.startswith("0_") else 1 for i in df.index]
	iterations = {column: 0 for column in df.columns}
	expected_amout = len(df.columns) * inp["dataset_per"] / 100
	write_log(inputs, f"Number of genes expected: {expected_amout}")
	pro_index = [index for index, elemento in enumerate(y) if elemento == 1]
	iterations = {column: 0 for column in df.columns}
	raw_values = {column: 0 for column in df.columns}
	attribute_names = df.columns
	for ind in range(0, inp['iterations']):
		write_log(inputs, f'Fitting Random Forest model Iteration {ind}.')
		rf_model = RandomForestClassifier(n_estimators=100)
		rf_model.fit(df, y)
		importances = rf_model.feature_importances_
		ind_importances = []
		for i in range(0, len(importances)):
			if importances[i] != 0:
				ind_importances.append(i)
		indices = np.argsort(importances)[::-1]
		explainer = shap.TreeExplainer(rf_model)
		shap_values = explainer.shap_values(df, check_additivity=False)
		raw_ite_values = {column: 0 for column in df.columns}
		for ind_gene in ind_importances:
			if shap_values[:, ind_gene, 1].mean() > 0:
				iterations[df.columns[ind_gene]] = iterations[df.columns[ind_gene]] + 1
				raw_ite_values[df.columns[ind_gene]] = shap_values[0, ind_gene, 1]
			elif shap_values[:, ind_gene, 1].mean() < 0:
				iterations[df.columns[ind_gene]] = iterations[df.columns[ind_gene]] - 1
				raw_ite_values[df.columns[ind_gene]] = shap_values[0, ind_gene, 1]
		coefficients = list(raw_ite_values.values())
		importance_df = pd.DataFrame({
			'locus_tag': attribute_names,
			'Coefficient': coefficients})
		df_coefficients_above = importance_df[importance_df["Coefficient"] > 0]
		df_coefficients_above.to_csv(f'{out_ite}rf_positive_{ind}.tsv', sep="\t", index=True)
		df_coefficients_below = importance_df[importance_df["Coefficient"] < 0]
		df_coefficients_below.to_csv(f'{out_ite}rf_negative_{ind}.tsv', sep="\t", index=True)
	df_iterations = pd.DataFrame(list(iterations.items()), columns=['locus_tag', 'occurrences'])
	df_iterations.to_csv(f"{out}_occurrences_raw.tsv", sep="\t", index=False)
	with open(f'{out}_positive.tsv', 'w') as output_file:
		output_file.write('locus_tag\toccurences\n')
		for locus in iterations:
			if iterations[locus] >= (inp["iterations"]*(inp["percentage"])/100):
				output_file.write(f'{locus}\t{iterations[locus]}\n')
	with open(f'{out}_negative.tsv', 'w') as output_file:
		output_file.write('locus_tag\toccurences\n')
		for locus in iterations:
			if iterations[locus] <= 0-(inp["iterations"]*(inp["percentage"])/100):
				output_file.write(f'{locus}\t{iterations[locus]}\n')
	write_log(inputs, 'Reading proteins fasta.')
	sequences = SeqIO.to_dict(SeqIO.parse(inp['faa'], "fasta"))
	extract_fasta(f'{out}_positive.tsv', sequences, "faa")
	extract_fasta(f'{out}_negative.tsv', sequences, "faa")
	del sequences
	write_log(inputs, 'Reading genes fasta.')
	sequences = SeqIO.to_dict(SeqIO.parse(inp['ffn'], "fasta"))
	extract_fasta(f'{out}_positive.tsv', sequences, "ffn")
	extract_fasta(f'{out}_negative.tsv', sequences, "ffn")
	del sequences
	plot_importance(list(iterations.values()),
				'Occurrences Obtained by Random Forest',
				(inp["iterations"]*(inp["percentage"])/100),
				(0 - inp["iterations"]*(inp["percentage"])/100),
				'Number of occurrences',
				out)

def run_eggnog(prefix, state):
	write_log(inputs, 'Running Eggnog-mapper.')
	if type(sh.which("emapper.py")) != str:
		write_log(inputs, f"Erro! Eggnog-mapper was not found in your environment. Please check!")
		sys.exit()
	else:
		cmd = f'emapper.py -i {prefix}_{state}.faa --override --cpu 15 -o {prefix}_{state} --excel'.split(' ')
		running = sbp.run(cmd)
		if running.returncode == 1:
			write_log(inputs, f'Erro while running: {cmd[2]}, all values will considered as Zero.')

def plot_kegg(inp, prefix):
	write_log(inputs, 'Ploting KEGG Annotation')
	kegg = pd.read_csv(inp["kegg"], sep="\t", index_col="Class4")
	try:
		positive = pd.read_excel(f'{prefix}_positive.emapper.annotations.xlsx', header=2, index_col="query").iloc[:-3]
	except:
		columns = ["query", "seed_ortholog", "evalue", "score", "eggNOG_OGs", "max_annot_lvl", "COG_category", "Description", "Preferred_name", "GOs", "EC", "KEGG_ko", "KEGG_Pathway", "KEGG_Module", "KEGG_Reaction", "KEGG_rclass", "BRITE", "KEGG_TC", "CAZy", "BiGG_Reaction", "PFAMs"]
		positive = pd.DataFrame(columns=columns)
		positive = positive.set_index("query")
	try:
		negative = pd.read_excel(f'{prefix}_negative.emapper.annotations.xlsx', header=2, index_col="query").iloc[:-3]
	except:
		columns = ["query", "seed_ortholog", "evalue", "score", "eggNOG_OGs", "max_annot_lvl", "COG_category", "Description", "Preferred_name", "GOs", "EC", "KEGG_ko", "KEGG_Pathway", "KEGG_Module", "KEGG_Reaction", "KEGG_rclass", "BRITE", "KEGG_TC", "CAZy", "BiGG_Reaction", "PFAMs"]
		negative = pd.DataFrame(columns=columns)
		negative = negative.set_index("query")
	unique_class1 = list(kegg["Class1"].unique())
	unique_class1.sort()
	unique_class2 = list(kegg["Class2"].unique())
	unique_class2.sort()
	total_positive = len(positive.index)
	total_negative = len(negative.index)
	dic = {}
	dic1 = {}
	dic2 = {}

	for class1 in unique_class1:
		dic1[class1] = 0
	for class2 in unique_class2:
		dic2[class2] = 0

	for elemento in positive["KEGG_ko"]:
		if ("," not in elemento) and (elemento != "-"):
			key = elemento
			if key.split(":")[1] not in kegg.index:
				continue
			if key not in dic.keys():
				dic[key] = 1
			else:
				dic[key] = dic[key] + 1
			c1 = list(kegg["Class1"])[list(kegg.index).index(key.upper().split(":")[1])]
			if c1 not in dic1.keys():
				dic1[c1] = 1
			else:
				dic1[c1] = dic1[c1] + 1
			c2 = list(kegg["Class2"])[list(kegg.index).index(key.upper().split(":")[1])]
			if c2 not in dic2.keys():
				dic2[c2] = 1
			else:
				dic2[c2] = dic2[c2] + 1
		elif "," in elemento:
			keys = elemento.split(",")
			for key in keys:
				if key.split(":")[1] not in kegg.index:
					continue
				if key not in dic.keys():
					dic[key] = 1
				else:
					dic[key] = dic[key] + 1
				c1 = list(kegg["Class1"])[list(kegg.index).index(key.upper().split(":")[1])]
				if c1 not in dic1.keys():
					dic1[c1] = 1
				else:
					dic1[c1] = dic1[c1] + 1
				c2 = list(kegg["Class2"])[list(kegg.index).index(key.upper().split(":")[1])]
				if c2 not in dic2.keys():
					dic2[c2] = 1
				else:
					dic2[c2] = dic2[c2] + 1

	positive_dic1 = dic1.copy()
	positive_dic2 = dic2.copy()

	dic = {}
	dic1 = {}
	dic2 = {}

	for class1 in unique_class1:
		dic1[class1] = 0
	for class2 in unique_class2:
		dic2[class2] = 0

	for elemento in negative["KEGG_ko"]:
		if ("," not in elemento) and (elemento != "-"):
			key = elemento
			if key.split(":")[1] not in kegg.index:
				continue
			if key not in dic.keys():
				dic[key] = 1
			else:
				dic[key] = dic[key] + 1
			c1 = list(kegg["Class1"])[list(kegg.index).index(key.upper().split(":")[1])]
			if c1 not in dic1.keys():
				dic1[c1] = 1
			else:
				dic1[c1] = dic1[c1] + 1
			c2 = list(kegg["Class2"])[list(kegg.index).index(key.upper().split(":")[1])]
			if c2 not in dic2.keys():
				dic2[c2] = 1
			else:
				dic2[c2] = dic2[c2] + 1
		elif "," in elemento:
			keys = elemento.split(",")
			for key in keys:
				if key.split(":")[1] not in kegg.index:
					continue
				if key not in dic.keys():
					dic[key] = 1
				else:
					dic[key] = dic[key] + 1
				c1 = list(kegg["Class1"])[list(kegg.index).index(key.upper().split(":")[1])]
				if c1 not in dic1.keys():
					dic1[c1] = 1
				else:
					dic1[c1] = dic1[c1] + 1
				c2 = list(kegg["Class2"])[list(kegg.index).index(key.upper().split(":")[1])]
				if c2 not in dic2.keys():
					dic2[c2] = 1
				else:
					dic2[c2] = dic2[c2] + 1

	negative_dic1 = dic1.copy()
	negative_dic2 = dic2.copy()

	index1 = []
	index2 = []
	for i in unique_class1:
		if (positive_dic1[i] == 0) and (negative_dic1[i] == 0):
			positive_dic1.pop(i)
			negative_dic1.pop(i)
		else:
			index1.append(i)
	for i in unique_class2:
		if (positive_dic2[i] == 0) and (negative_dic2[i] == 0):
			positive_dic2.pop(i)
			negative_dic2.pop(i)
		else:
			index2.append(i)

	positive_values1 = list(positive_dic1.values())
	positive_values2 = list(positive_dic2.values())
	negative_values1 = list(negative_dic1.values())
	negative_values2 = list(negative_dic2.values())

	for i in range(0, len(positive_values1)):
		if total_positive != 0:
			positive_values1[i] = (positive_values1[i]/total_positive) * 100
		else:
			positive_values1[i] = 0

	for i in range(0, len(positive_values2)):
		if total_positive != 0:
			positive_values2[i] = (positive_values2[i]/total_positive) * 100
		else:
			positive_values2[i] = 0

	for i in range(0, len(negative_values1)):
		if total_negative != 0:
			negative_values1[i] = (negative_values1[i]/total_negative) * 100
		else:
			negative_values1[i] = 0

	for i in range(0, len(negative_values2)):
		if total_negative != 0:
			negative_values2[i] = (negative_values2[i]/total_negative) * 100
		else:
			negative_values2[i] = 0

	df1 = pd.DataFrame({"Category":index1, "Positive":positive_values1, "Negative":negative_values1}).set_index("Category")
	df1.to_csv(f'{prefix}_kegg_df_1.tsv', sep='\t', index=True)
	df2 = pd.DataFrame({"Category":index2, "Positive":positive_values2, "Negative":negative_values2}).set_index("Category")
	df2.to_csv(f'{prefix}_kegg_df_2.tsv', sep='\t', index=True)

	# Plotar o gráfico de barras
	plt.clf()
	df1.plot(kind='bar', figsize=(10, 6), color=['#2E8B57', "crimson"])
	plt.xlabel('Category')
	plt.ylabel('Genes percentage')
	plt.title('Genes by KEGG category')
	plt.xticks(rotation=45)  # Rotacionar os rótulos do eixo x para melhor visualização
	plt.legend(title='Dataset')
	plt.grid(axis='y', linestyle='--', alpha=0.5)  # Adicionar linhas de grade no eixo y
	plt.tight_layout()  # Ajustar layout
	plt.savefig(f"{prefix}_kegg_plot_1.png", dpi=600)

	# Plotar o gráfico de barras
	plt.clf()
	df2.plot(kind='bar', figsize=(15, 12), color=['#2E8B57', "crimson"])
	plt.xlabel('Category')
	plt.ylabel('Genes percentage')
	plt.title('Genes by KEGG category')
	plt.xticks(rotation=90)  # Rotacionar os rótulos do eixo x para melhor visualização
	plt.legend(title='Dataset')
	plt.grid(axis='y', linestyle='--', alpha=0.5)  # Adicionar linhas de grade no eixo y
	plt.tight_layout()  # Ajustar layout
	plt.savefig(f"{prefix}_kegg_plot_2.png", dpi=600)

def plot_cog(inp, prefix):
	write_log(inputs, 'Ploting COG Annotation')
	dic_cog = pd.read_csv(inp["cog"], sep="\t", index_col="Abbreviation")
	try:
		positive = pd.read_excel(f'{prefix}_positive.emapper.annotations.xlsx', header=2, index_col="query").iloc[:-3]
	except:
		columns = ["query", "seed_ortholog", "evalue", "score", "eggNOG_OGs", "max_annot_lvl", "COG_category", "Description", "Preferred_name", "GOs", "EC", "KEGG_ko", "KEGG_Pathway", "KEGG_Module", "KEGG_Reaction", "KEGG_rclass", "BRITE", "KEGG_TC", "CAZy", "BiGG_Reaction", "PFAMs"]
		positive = pd.DataFrame(columns=columns)
		positive = positive.set_index("query")
	try:
		negative = pd.read_excel(f'{prefix}_negative.emapper.annotations.xlsx', header=2, index_col="query").iloc[:-3]
	except:
		columns = ["query", "seed_ortholog", "evalue", "score", "eggNOG_OGs", "max_annot_lvl", "COG_category", "Description", "Preferred_name", "GOs", "EC", "KEGG_ko", "KEGG_Pathway", "KEGG_Module", "KEGG_Reaction", "KEGG_rclass", "BRITE", "KEGG_TC", "CAZy", "BiGG_Reaction", "PFAMs"]
		negative = pd.DataFrame(columns=columns)
		negative = negative.set_index("query")
	letras = list(dic_cog.index)
	dic_p = {letra: 0 for letra in letras}
	dic_n = {letra: 0 for letra in letras}
	total_positive = len(positive.index)
	total_negative = len(negative.index)
	for letra in letras:
		index = 0
		for i in positive["COG_category"]:
			if letra in i:
				index = index + 1
		dic_p[letra] = index
	for letra in letras:
		index = 0
		for i in negative["COG_category"]:
			if letra in i:
				index = index + 1
		dic_n[letra] = index

	# Converter os dicionários em DataFrames
	df1 = pd.DataFrame(list(dic_p.items()))[1]
	soma = sum(list(df1))
	lista = []
	for i in list(df1):
		try:
			temp = (i/soma)*100
			lista.append(temp)
		except:
			lista.append(0)
	df1 = lista

	df2 = pd.DataFrame(list(dic_n.items()))[1]
	soma = sum(list(df2))
	lista = []
	for i in list(df2):
		try:
			temp = (i/soma)*100
			lista.append(temp)
		except:
			lista.append(0)
	df2 = lista
	df = pd.DataFrame({"Category":list(dic_cog["COG Categories"]), "Positive":list(df1), "Negative":list(df2)}).set_index("Category")
	df.to_csv(f'{prefix}_cog_df_2.tsv', sep='\t', index=True)

	# Plotar o gráfico de barras
	plt.clf()
	df.plot(kind='bar', figsize=(14, 12), color=['#2E8B57', "crimson"])
	plt.xlabel('Category')
	plt.ylabel('Percentage of genes')
	plt.title('Genes by COG category')
	plt.xticks(rotation=90)  # Rotacionar os rótulos do eixo x para melhor visualização
	plt.legend(title='Dataset')
	plt.grid(axis='y', linestyle='--', alpha=0.5)  # Adicionar linhas de grade no eixo y
	plt.tight_layout()  # Ajustar layout
	plt.savefig(f'{prefix}_cog_plot_2.png', dpi=600)

	wide_cat_p = {"Cellular component": 0,
				  "Cellular processes and sinaling": 0,
				  "Information storage and processing": 0,
				  "Metabolism": 0,
				  "Poorly characterized": 0}
	for i in dic_p:
		if dic_cog["Function"][i] not in wide_cat_p:
			wide_cat_p[dic_cog["Function"][i]] = 0
		wide_cat_p[dic_cog["Function"][i]] = wide_cat_p[dic_cog["Function"][i]] + dic_p[i]
	for i in wide_cat_p:
		if total_positive != 0:
			wide_cat_p[i] = wide_cat_p[i] / total_positive *100
		else:
			wide_cat_p[i] = 0

	wide_cat_n = {"Cellular component": 0,
				  "Cellular processes and sinaling": 0,
				  "Information storage and processing": 0,
				  "Metabolism": 0,
				  "Poorly characterized": 0}
	for i in dic_n:
		if dic_cog["Function"][i] not in wide_cat_n:
			wide_cat_n[dic_cog["Function"][i]] = 0
		wide_cat_n[dic_cog["Function"][i]] = wide_cat_n[dic_cog["Function"][i]] + dic_n[i]
	for i in wide_cat_n:
		if total_negative != 0:
			wide_cat_n[i] = wide_cat_n[i] / total_negative *100
		else:
			wide_cat_n[i] = 0

	df = pd.DataFrame({"Category":list(wide_cat_p.keys()), "Positive":list(wide_cat_p.values()), "Negative":list(wide_cat_n.values())}).set_index("Category")
	df.to_csv(f'{prefix}_cog_df_1.tsv', sep='\t', index=True)

	# Plotar o gráfico de barras
	plt.clf()
	df.plot(kind='bar', figsize=(12, 10), color=['#2E8B57', "crimson"])
	plt.xlabel('Category')
	plt.ylabel('Percentage of genes')
	plt.title('Genes by COG category')
	plt.xticks(rotation=45)  # Rotacionar os rótulos do eixo x para melhor visualização
	plt.legend(title='Dataset')
	plt.grid(axis='y', linestyle='--', alpha=0.5)  # Adicionar linhas de grade no eixo y
	plt.tight_layout()  # Ajustar layout
	plt.savefig(f'{prefix}_cog_plot_1.png', dpi=600)

if __name__ == '__main__':
	inputs = arguments()
	prefixes = []
	if inputs['lr'] == True:
		run_lr(inputs, 'alpha_values')
		prefixes.append(f'{inputs["outdir"]}{os.sep}alpha_values')
		for j in ['positive', 'negative']:
			run_eggnog(f'{inputs["outdir"]}{os.sep}alpha_values', j)
		plot_kegg(inputs, f'{inputs["outdir"]}{os.sep}alpha_values')
		plot_cog(inputs, f'{inputs["outdir"]}{os.sep}alpha_values')
	if inputs['svm'] == True:
		run_svm(inputs, 'vector_values')
		prefixes.append(f'{inputs["outdir"]}{os.sep}vector_values')
		for j in ['positive', 'negative']:
			run_eggnog(f'{inputs["outdir"]}{os.sep}vector_values', j)
		plot_kegg(inputs, f'{inputs["outdir"]}{os.sep}vector_values')
		plot_cog(inputs, f'{inputs["outdir"]}{os.sep}vector_values')
	if inputs['rf'] == True:
		run_rf(inputs, 'random_forest_values')
		prefixes.append(f'{inputs["outdir"]}{os.sep}random_forest_values')
		for j in ['positive', 'negative']:
			run_eggnog(f'{inputs["outdir"]}{os.sep}random_forest_values', j)
		plot_kegg(inputs, f'{inputs["outdir"]}{os.sep}random_forest_values')
		plot_cog(inputs, f'{inputs["outdir"]}{os.sep}random_forest_values')
	if prefixes == []:
		sh.rmtree(inputs['outdir'])
		raise Exception('No analysis performed. Please select AT LEAST one method and run again.')
