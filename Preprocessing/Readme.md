# Preprocessing for datasets B,D and E

To follow our steps for preprocessing dataset B,D and E, you need to clone this repository and run a terminal in the Pre-processing directory.

## Build docker container

    docker build -t sl-preprocessing:1.0 .

## Download reference

    mkdir reference
    wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_33/GRCh38.primary_assembly.genome.fa.gz
    wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_33/gencode.v33.primary_assembly.annotation.gtf.gz

Unpack the files and place them as genome.fa and annotation.gtf in the reference subfolder

## Prepare STAR index

Run the docker-container 
    docker run -it --rm -v ./index:/index -v ./reference:/reference sl-preprocessing bash

Then, inside the container, generate the index 
    STAR --runMode genomeGenerate --genomeDir /index --genomeFastaFiles /reference/genome.fa --runThreadN 32


## Modify docker-compose.yml

Check the directories listed in the docker-compose file. Place your fastq-files in the fastq subfolder. The Snakefile expects the following filename structure: {sample}_S{sid}_L{lane}_R{read}_{num}.fastq.gz

## Run pipeline

    docker-compose up

Upon completion, you can find the results in the output folder.

