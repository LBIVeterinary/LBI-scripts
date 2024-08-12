#!/usr/bin/env python3
import sys
import multiprocessing as mtp
import pandas as pd
import re, random, string, gzip, os
import urllib.request as urlget
import shutil as sh
from datetime import datetime
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def download_genomes(par):
	attempts = 1
	cmds = []
	while attempts < 6:
		print(f"Start download attempt: {attempts} - {par[7]} {par[3]} {par[4]} {par[5]}")
		try:
			download = urlget.urlretrieve(par[0])[0]
			with gzip.open(download, 'rb') as f_in:
				with open(par[2], 'wb') as f_out:
					sh.copyfileobj(f_in, f_out)
			print(f"File {par[2]} was downloaded succesfully...")
			os.remove(download)
			cmd = f"prokka --addgenes --force --species {par[4]} --genus {par[3]} --strain {par[5]} {par[2]} --prefix {par[6]} --outdir {par[6]} --locustag {par[6]}\n"
			cmds.append(cmd)
			break
		except:
			attempts = attempts + 1
			if attempts == 6:
				print(f"It was not possible to download file {par[2]} from BioSample {par[7]}. Please check your inputs.")
	if par[9] == True:
		with open("PROKKA.sh", "a") as prokka_file:
			for commands in cmds:
				prokka_file.write(commands)

def getNCBI(file, file_type, cpus):
	if file_type == "gbff":
		file_ext = "_genomic.gbff.gz"
	elif file_type == "fna":
		file_ext = "_genomic.fna.gz"
	elif file_type == "faa":
		file_ext = "_protein.faa.gz"
	csv = pd.read_csv(file, sep="\t", index_col="Assembly Accession")
	check_file_dup = []
	args = []
	directory = "Genomes_"+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
	os.mkdir(directory)
	for i in csv.index:
		genus = csv["Organism Name"][i].split(" ")[0]
		epitet = csv["Organism Name"][i].split(" ")[1]
		if type(csv["Organism Infraspecific Names Strain"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Strain"][i])
		elif type(csv["Organism Infraspecific Names Isolate"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Isolate"][i])
		elif type(csv["Organism Infraspecific Names Breed"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Breed"][i])
		elif type(csv["Organism Infraspecific Names Cultivar"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Cultivar"][i])
		elif type(csv["Organism Infraspecific Names Ecotype"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Ecotype"][i])
		else:
			print(f"It wasn't possible to find strain name for file {i}")
			exit()
		assembly = csv["Assembly Name"][i]
		biosample = csv["Assembly BioSample Accession"][i]
		if strain not in check_file_dup:
			check_file_dup.append(strain)
		else:
			strain = f"{strain}_dup_{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}"
			check_file_dup.append(strain)
		file_out = f"{directory}{os.sep}{genus[0]}{epitet}_{strain}.{file_type}"
		locustag = f"{genus[0]}{epitet[0]}_{strain}"
		access = re.findall("...",i.replace("_", ""))
		ftp_link = f"ftp://ftp.ncbi.nlm.nih.gov/genomes/all/"
		for tri in access:
			ftp_link = f"{ftp_link}{tri}/"
		ftp_link = f"{ftp_link}{i}_{assembly}/{i}_{assembly}{file_ext}"
		file_inp = f"{i}_{assembly}{file_ext}"
		if file_type == "fna":
			arg = (ftp_link, file_inp, file_out, genus, epitet, strain, locustag, i, directory, True)
		else:
			arg = (ftp_link, file_inp, file_out, genus, epitet, strain, locustag, i, directory, False)
		args.append(arg)
	if ("PROKKA.sh" in os.listdir()) and (file_type == "fna"):
		os.remove("PROKKA.sh")
	if cpus >= mtp.cpu_count():
		cpus = int(mtp.cpu_count()) - 1
	print(f"Using {cpus} threads.\n")
	pool = mtp.Pool(processes=cpus)
	pool.map(download_genomes,args)
	pool.close()
	pool.join()

def download_metadata(par):
	print(f"Evaluating metadata from {par[1]} - {par[3]} {par[4]} {par[5]}")
	download = urlget.urlretrieve(par[0])[0]
	html = open(download, 'rt')
	txt = html.readlines()
	html.close()
	os.remove(download)
	x = ""
	for j in txt:
		temp = j.strip().replace("\n", "")
		x = x + temp
	if x.find("<th>host</th><td>") != -1:
		host = (x[(x.find("<th>host</th><td>")+17):(x.find("<", (x.find("<th>host</th><td>")+17)))]).strip().capitalize()
	else:
		host = "NA"
	if x.find("host disease</th><td>") != -1:
		disease = (x[(x.find("host disease</th><td>")+21):(x.find("<", (x.find("host disease</th><td>")+21)))]).strip().capitalize()
	else:
		disease = "NA"
	if x.find("geo_loc_name=") != -1:
		geo = (x[(x.find("geo_loc_name=")+13):(x.find("&", (x.find("geo_loc_name=")+13)))]).strip()
	else:
		geo = "NA"
	if x.find(">isolation source</th><td>") != -1:
		source = (x[(x.find(">isolation source</th><td>")+26):(x.find("<", (x.find(">isolation source</th><td>")+26)))]).strip().capitalize()
	else:
		source = "NA"
	line = f"{par[6]}\t{par[3]} {par[4]}\t{par[5]}\t{par[1]}\t{host}\t{disease}\t{source}\t{geo}\n"
	file = open(par[7], "a")
	file.write(line)
	file.close()

def getMETA(file, cpus):
	csv = pd.read_csv(file, sep="\t", index_col="Assembly Accession")
	parameters = []
	file_name = f"Metadata_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.tsv"
	file = open(file_name, "w")
	file.write("Genome Accession\tSpecies\tStrain\tBioSample\tHost\tDisease\tIsolation Source\tGeographic Localization\n")
	file.close()
	for i in csv.index:
		genus = csv["Organism Name"][i].split(" ")[0]
		epitet = csv["Organism Name"][i].split(" ")[1]
		if type(csv["Organism Infraspecific Names Strain"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Strain"][i])
		elif type(csv["Organism Infraspecific Names Isolate"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Isolate"][i])
		elif type(csv["Organism Infraspecific Names Breed"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Breed"][i])
		elif type(csv["Organism Infraspecific Names Cultivar"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Cultivar"][i])
		elif type(csv["Organism Infraspecific Names Ecotype"][i]) == str:
			strain = re.sub("((?![\.A-z0-9_-]).)", "_", csv["Organism Infraspecific Names Ecotype"][i])
		else:
			print(f"It wasn't possible to find strain name for file {i}")
			exit()
		assembly = csv["Assembly Name"][i]
		biosample = csv["Assembly BioSample Accession"][i]
		link = f"https://www.ncbi.nlm.nih.gov/biosample/{biosample}"
		parameters.append((link, biosample, assembly, genus, epitet, strain, i, file_name))
	if cpus >= mtp.cpu_count():
		cpus = int(mtp.cpu_count()) - 1
	print(f"Using {cpus} threads.\n")
	pool = mtp.Pool(processes=cpus)
	pool.map(download_metadata,parameters)
	pool.close()
	pool.join()
	file.close()
	return(file_name)

def generate_map(input_file):
	latlon = urlget.urlretrieve("https://raw.githubusercontent.com/dlnrodrigues/panvita/dlnrodrigues-Supplementary/latlon.csv", "latlon.csv")[0]
	meta = pd.read_csv(input_file, sep="\t")
	countries = meta["Geographic Localization"].tolist()
	latlon = "latlon.csv"
	try:
		countries_keys = pd.read_csv(latlon, sep=";", index_col="homecontinent")
	except:
		print("It was not possible to open latlon.csv file.")
		exit()
	unique = []
	k = []
	for i in countries:
		if type(i) == str:
			if ":" not in i:
				string = i
				k.append(string)
			else:
				string = i.split(":")[0]
				k.append(string)
			if string not in unique:
				unique.append(string)
	cont = []
	for i in unique:
		cont.append(k.count(i))
	data = {}
	homelat = []
	homelon = []
	cont = []
	temp_unique = []
	for i in unique:
		if i in countries_keys.index.values.tolist():
			homelat.append(countries_keys["homelat"][i])
			homelon.append(countries_keys["homelon"][i])
			cont.append(k.count(i))
			temp_unique.append(i)
		else:
			print(f"Unidentified country: {i}")
	data = {"homecontinent": temp_unique,
			"homelat": homelat,
			"homelon": homelon,
			"n": cont}
	data = pd.DataFrame.from_dict(data)
	file_name = f"Metadata_countries_count_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.tsv"
	data.to_csv(file_name, sep="\t")
	plt.figure(figsize=(20, 15))
	plt.rcParams["figure.figsize"]=20,15;
	m=Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=80)
	m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
	m.fillcontinents(color='green', alpha=0.3)
	m.drawcoastlines(linewidth=0.1, color="white")
	data['labels_enc'] = pd.factorize(data['homecontinent'])[0]
	m.scatter(
		x=data['homelon'], 
		y=data['homelat'], 
		s=data['n']*10, 
		alpha=0.4, 
		c=data['labels_enc'], 
		cmap="plasma")
	#plt.text(-175, -62,'Isolates geographical localization', ha='left', va='bottom', size=, color='#555555' )
	file_name = f"Metadata_map_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.png"
	plt.savefig(file_name, dpi=600, bbox_inches="tight")
	os.remove(latlon)

help_msg = f'''
Dear user,
This is a short script to download genomes, proteomes and metadatas from NCBI page.
To use this tool you'll need to provide a TSV table obteined from NCBI Genome page:

https://www.ncbi.nlm.nih.gov/datasets/genome/

USAGE:
{sys.argv[0]} <arg> <file.tsv>

--genome	Download genome file (fna)
--proteome	Download proteome file (faa)
--genbank	Download genbank file (gbf)
--metadata	Download metadata from a tsv file and plot a map
--cpu		Set the number of desired threads (default = 1)
--help		Print this message

WARNING!!!
The file extension MUST be .tsv or .csv. Any other type of file will not be accepted!
'''

if (("--genome" not in sys.argv) and ("--proteome" not in sys.argv) and ("--genbank" not in sys.argv) and ("--metadata" not in sys.argv)) or (("-h" in sys.argv) or ("--help" in sys.argv)):
	print(help_msg)
	exit()

inputs = []
for i in sys.argv:
	if i.endswith(".tsv") or i.endswith(".csv"):
		inputs.append(i)

if inputs == []:
	print(help_msg)
	exit()

threads = 1
if "--cpu" in sys.argv:
	try:
		threads = int(sys.argv[sys.argv.index("--cpu") + 1])
	except:
		print(f"--cpu must be an integer number *** {sys.argv[sys.argv.index('--cpu') + 1]} *** Verify!")
		exit()

for csv in inputs:
	if "--genome" in sys.argv:
		getNCBI(csv, "fna", threads) # Accepts gbff, faa, fna
	if "--proteome" in sys.argv:
		getNCBI(csv, "faa", threads) # Accepts gbff, faa, fna
	if "--genbank" in sys.argv:
		getNCBI(csv, "gbff", threads) # Accepts gbff, faa, fna
	if "--metadata" in sys.argv:
		meta_file = getMETA(csv, threads)
		generate_map(meta_file)
