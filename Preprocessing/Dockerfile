FROM continuumio/miniconda3:4.7.12

LABEL maintainer="Matthias Becker"

RUN conda config --add channels defaults
RUN conda config --add channels conda-forge
RUN conda config --add channels bioconda

RUN conda install -y snakemake=5.9.1 star=2.7.3a multiqc=1.8 samtools=1.9 ncurses=6.1 fastqc=0.11.9-0

ADD Snakefile /

ENV JOBS 32

CMD snakemake --snakefile /Snakefile --jobs $JOBS


