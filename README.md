Forked from: https://github.com/abisee/cnn-dailymail and https://github.com/chmille3/process_data_for_pointer_summrizer

**NOTE:** This code is in Python 3 as compared to the original code which was in Python 2.

This code produces a dataset compatible with code for the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)*. It processes the dataset into the binary format expected by the [code](https://github.com/atulkum/pointer_summarizer) for the PyTorch model. I use my own fork of the code [here](https://github.com/ashishbaghudana/pointer_summarizer)

**NOTE:** If you are looking to process the CNN-Dailymail dataset, please follow the instructions on https://github.com/abisee/cnn-dailymail or https://github.com/becxer/cnn-dailymail. The `make_datafiles.py` script in this repository is heavily modified for processing dataset in a different format.

## Expected Dataset Format
The `make_datafiles.py` script expects data in two directories as follows.

```
dataset/
├── fulltext/
│   ├── 0.text
│   ├── 1.text
│   └── ...
└── summary/
    ├── 0.summary
    ├── 1.summary
    └── ...
```

Each `*.text` file contains the full text of the document, and the corresponding `*.summary` file contains the summary. The number of documents in the `fulltext` directory and `summary` directory must match.

## Pre-requisites
The `make_datafiles.py` requires [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) installed for preprocessing the dataset. Use the `setup_stanford_corenlp.sh` script to setup these dependencies as follows.

```bash
source ./setup_stanford_corenlp.sh
echo $CLASSPATH
# export CLASSPATH=$(pwd)/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
```

**NOTE:** The `source` command is important as the last step of the script exports the `CLASSPATH`. If `echo $CLASSPATH` returns empty, the setup was not successful.

## Instructions to run preprocessing

The `make_datafiles.py` comes with usage information that can be activated using the `-h` flag.

```bash
$ python make_datafiles.py -h

usage: Preprocess dataset for Pointer Generator Networks [-h] -f FULLTEXT -s
                                                         SUMMARY -o OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -f FULLTEXT, --fulltext FULLTEXT
                        Path to directory containing full text documents
  -s SUMMARY, --summary SUMMARY
                        Path to directory containing summaries
  -o OUTPUT, --output OUTPUT
                        Path to directory to contain the .bin files
```

Run this script by pointing it to the full text and summary directories as follows:

```bash
$ python make_datafiles.py -f dataset/fulltext -s dataset/summary -o output/
```

The script will create three directories within the `output` folder as follows:

```
dataset/
├── tokenized_documents/
│   ├── 0.text
│   ├── 1.text
│   ├── ...
│   └── n.text
├── tokenized_summaries/
│   ├── 0.summary
│   ├── 1.summary
│   ├── ...
│   └── n.summary
└── finished_files/
    ├── train.bin
    ├── test.bin
    ├── val.bin
    ├── vocab
    └── chunked
        ├── train_000.bin
        ├── ...
        ├── train_00x.bin
        ├── test_000.bin
        ├── ...
        ├── test_00x.bin
        ├── val_000.bin
        ├── ...
        └── val_00x.bin
```
