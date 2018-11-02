import collections
import os
import random
import struct
import subprocess
from argparse import ArgumentParser
from glob import glob

from nltk import sent_tokenize
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

TOKENIZED_DOCUMENTS_DIR = "tokenized_documents"
TOKENIZED_SUMMARY_DIR = "tokenized_summaries"
FINISHED_FILES_DIR = "finished_files"
CHUNKS_DIR = os.path.join(FINISHED_FILES_DIR, "chunked")

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data
SEED = 42


def chunk_file(set_name):
    in_file = os.path.join(FINISHED_FILES_DIR, '%s.bin' % set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(
            CHUNKS_DIR, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len,
                                            reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(CHUNKS_DIR):
        os.mkdir(CHUNKS_DIR)
    # Chunk the data
    # for set_name in ['train', 'val', 'test']:
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % CHUNKS_DIR)


def get_documents(directory,
                  extension=None,
                  split=False,
                  split_percentage=(0.7, 0.15, 0.15)):
    if extension is None:
        extension = ''
    if not split:
        return [
            os.path.basename(p)
            for p in glob(os.path.join(directory, extension))
        ]
    else:
        docs = [
            os.path.basename(p)
            for p in glob(os.path.join(directory, extension))
        ]
        random.seed(SEED)
        random.shuffle(docs)

        train_start = 0
        train_end = train_start + int(split_percentage[0] * len(docs))

        test_start = train_end + 1
        test_end = test_start + int(split_percentage[1] * len(docs))

        val_start = test_end + 1
        val_end = len(docs)

        return (docs[train_start:train_end], docs[test_start:test_end],
                docs[val_start:val_end])


def tokenize_documents(fulltext_dir, summary_dir, tokenized_documents_dir,
                       tokenized_summary_dir, tokenize_docs,
                       tokenize_summaries):
    """
    Maps a whole directory of .story files to a tokenized version using
    Stanford CoreNLP Tokenizer
    """
    print("Preparing to tokenize %s, %s to %s..." % (fulltext_dir, summary_dir,
                                                     tokenized_documents_dir))
    documents = get_documents(fulltext_dir, "*.text")
    summaries = get_documents(summary_dir, "*.summary")
    print("Making list of files to tokenize...")

    with open("mapping_fulltext.txt", "w") as f:
        for s in documents:
            f.write("%s \t %s\n" % (os.path.join(fulltext_dir, s),
                                    os.path.join(tokenized_documents_dir, s)))

    with open("mapping_summary.txt", "w") as f:
        for s in summaries:
            f.write("%s \t %s\n" % (os.path.join(summary_dir, s),
                                    os.path.join(tokenized_summary_dir, s)))

    command_fulltext = [
        'java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList',
        '-preserveLines', 'mapping_fulltext.txt'
    ]
    command_summary = [
        'java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList',
        '-preserveLines', 'mapping_summary.txt'
    ]

    if tokenize_docs:
        print("Tokenizing %i fulltext documents in %s and saving in %s..." %
              (len(documents), fulltext_dir, tokenized_documents_dir))
        subprocess.call(command_fulltext)

    if tokenize_summaries:
        print("Tokenizing %i summary documents in %s and saving in %s..." %
              (len(documents), summary_dir, tokenized_summary_dir))
        subprocess.call(command_summary)

    print("Stanford CoreNLP Tokenizer has finished.")

    # Remove temporary files
    os.remove("mapping_fulltext.txt")
    os.remove("mapping_summary.txt")

    # Check that the tokenized stories directory contains the same number of
    # files as the original directory
    num_orig = len(documents)
    num_tokenized = len(get_documents(tokenized_documents_dir, "*.text"))
    if num_orig != num_tokenized:
        raise Exception(
            ("The tokenized documents directory %s contains %i files, but it "
             + "should contain the same number as %s (which has %i files). " +
             "Was there an error during tokenization?") %
            (tokenized_documents_dir, num_tokenized, fulltext_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" %
          (fulltext_dir, tokenized_documents_dir))

    num_orig = len(summaries)
    num_tokenized = len(get_documents(tokenized_summary_dir, "*.summary"))
    if num_orig != num_tokenized:
        raise Exception(
            ("The tokenized summaries directory %s contains %i files, but it "
             + "should contain the same number as %s (which has %i files). " +
             "Was there an error during tokenization?") %
            (tokenized_summary_dir, num_tokenized, summary_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" %
          (summary_dir, tokenized_summary_dir))


def read_text_file(text_file):
    """Read a text file and tokenize to sentences"""
    with open(text_file, "r") as f:
        lines = sent_tokenize(f.read())
    return lines


def get_fulltext_summary(fulltext_doc, summary_doc):
    # Article lines
    fulltext_lines = [
        line.lower() for line in read_text_file(
            os.path.join(TOKENIZED_DOCUMENTS_DIR, fulltext_doc))
    ]

    # Summary lines
    summary_lines = [
        line.lower() for line in read_text_file(
            os.path.join(TOKENIZED_SUMMARY_DIR, summary_doc))
    ]

    # Make article into a single string
    fulltext = ' '.join(fulltext_lines)

    # Make abstract into a signle string, putting <s> and </s> tags
    # around the sentences
    summary = ' '.join([
        "%s %s %s" % (SENTENCE_START, sent, SENTENCE_END)
        for sent in summary_lines
    ])

    return fulltext, summary


def write_to_bin(out_file, fulltext_docs, summary_docs, makevocab=False):
    """
    Reads the tokenized .text files corresponding to the urls listed in the
    url_file and writes them to a out_file.
    """
    num_docs = len(fulltext_docs)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, (f_doc, s_doc) in enumerate(zip(fulltext_docs, summary_docs)):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" %
                      (idx, num_docs, float(idx) * 100.0 / float(num_docs)))

            # Get the strings to write to .bin file
            article, abstract = get_fulltext_summary(f_doc, s_doc)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend(
                [article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend(
                [abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [
                    t for t in abs_tokens
                    if t not in [SENTENCE_START, SENTENCE_END]
                ]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(FINISHED_FILES_DIR, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    parser = ArgumentParser(
        "Preprocess dataset for Pointer Generator Networks")

    paths = parser.add_argument_group('Path arguments')
    paths.add_argument(
        "-f",
        "--fulltext",
        help="Path to directory containing full text documents",
        required=True)
    paths.add_argument(
        "-s",
        "--summary",
        help="Path to directory containing summaries",
        required=True)
    paths.add_argument(
        "-o",
        "--output",
        help="Path to directory to contain the .bin files",
        required=True)

    actions = parser.add_mutually_exclusive_group("Set of actions")
    actions.add_argument(
        "--only_tokenize_fulltext",
        help="Only tokenize full text documents",
        action="store_true")
    actions.add_argument(
        "--only_tokenize_summaries",
        help="Only tokenize summary documents",
        action="store_true")
    actions.add_argument(
        "--write_to_bin", help="Write as bin files", action="store_true")
    actions.add_argument(
        "--all", help="Perform all actions", action="store_true")

    args = parser.parse_args()

    # get the number of stories to process
    num_expected_stories = len(os.listdir(args.fulltext))

    TOKENIZED_DOCUMENTS_DIR = os.path.join(args.output,
                                           TOKENIZED_DOCUMENTS_DIR)
    TOKENIZED_SUMMARY_DIR = os.path.join(args.output, TOKENIZED_SUMMARY_DIR)
    FINISHED_FILES_DIR = os.path.join(args.output, FINISHED_FILES_DIR)
    CHUNKS_DIR = os.path.join(args.output, CHUNKS_DIR)

    print("Number of stories: ", num_expected_stories)

    # Create some new directories
    if not os.path.exists(TOKENIZED_DOCUMENTS_DIR):
        os.makedirs(TOKENIZED_DOCUMENTS_DIR, exist_ok=True)
    if not os.path.exists(TOKENIZED_SUMMARY_DIR):
        os.makedirs(TOKENIZED_SUMMARY_DIR, exist_ok=True)
    if not os.path.exists(FINISHED_FILES_DIR):
        os.makedirs(FINISHED_FILES_DIR, exist_ok=True)

    if args.all:
        args.only_tokenize_fulltext = True
        args.only_tokenize_summaries = True
        args.write_to_bin = True

    if args.only_tokenize_fulltext or args.only_tokenize_summaries:
        # Run Stanford Tokenizer on every full text document
        tokenize_documents(args.fulltext, args.summary,
                           TOKENIZED_DOCUMENTS_DIR, TOKENIZED_SUMMARY_DIR,
                           args.only_tokenize_fulltext,
                           args.only_tokenize_summaries)

    fulltext_docs_train, fulltext_docs_test, fulltext_docs_val = get_documents(
        TOKENIZED_DOCUMENTS_DIR, "*.text", split=True)
    summary_docs_train, summary_docs_test, summary_docs_val = get_documents(
        TOKENIZED_SUMMARY_DIR, "*.summary", split=True)

    if args.write_to_bin:
        write_to_bin(
            os.path.join(FINISHED_FILES_DIR, "train.bin"),
            fulltext_docs_train,
            summary_docs_train,
            makevocab=True)
        write_to_bin(
            os.path.join(FINISHED_FILES_DIR, "test.bin"),
            fulltext_docs_test,
            summary_docs_test,
            makevocab=False)
        write_to_bin(
            os.path.join(FINISHED_FILES_DIR, "val.bin"),
            fulltext_docs_val,
            summary_docs_val,
            makevocab=False)
        chunk_all()
