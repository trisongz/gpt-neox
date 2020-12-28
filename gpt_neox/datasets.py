import torch
from torch.utils.data import Dataset, IterableDataset
from .data_utils import get_tokenizer, natural_sort, skip, FixedSizeOrderedDict
import random
import glob
import tensorflow as tf
import re
import logging
from itertools import cycle, chain
from transformers import GPT2Tokenizer
from gpt_neox.the_pile import ThePile
import tensorflow_datasets as tfds

class GPT2Dataset(Dataset):

    def __init__(self, glob_pattern, seq_len, seed=1, shuffle_input_filenames=True, pretokenized=True,
                 filetype="tfrecords", mode="chunks", train=True, tokenizer=None, **kwargs):

        super().__init__()
        self.files = glob.glob(glob_pattern)  # glob pattern pointing to files
        self.seed = seed  # random seed for shuffling

        # shuffle or sort files
        if shuffle_input_filenames:
            random.seed(self.seed)
            random.shuffle(self.files)
        else:
            self.files = natural_sort(self.files)
        self.filetype = filetype  # filetype ["tfrecords"]
        implemented_filetypes = ["tfrecords"]
        if self.filetype not in implemented_filetypes:
            raise NotImplementedError

        self.processed_files = FixedSizeOrderedDict(max=2)  # storage for lazily loading data

        # parses the length of the files, either by encoding in the filenames or by iterating over them
        self._get_lens()

        self.seq_len = seq_len  # set sequence length
        self.mode = mode  # set mode ["chunks"]
        implemented_modes = ["chunks"]
        if self.mode not in implemented_modes:
            raise NotImplementedError

        self.pretokenized = pretokenized
        if not self.pretokenized:
            raise NotImplementedError  # TODO: tokenize text data on the fly

        self.train = train

    def _get_number_of_documents(self, filename):
        # extracts number of files from a filename formatted "<name>_<num_documents>.{filetype}."
        # if no pattern is matched, returns None
        match = re.search("_(\d{1,})." + self.filetype + "$", filename)
        return int(match.group(1)) if match is not None else match

    def _get_number_of_documents_by_iteration(self, filename):
        # extracts number of files from a tfrecord document in the event it doesn't have metadata in the filename
        # this could be very slow.
        logging.warning(
            "Found no metadata found in filename - iterating through first tfrecord to find global length")
        count = 0
        if self.filetype == "tfrecords":
            for _ in tf.io.tf_record_iterator(filename):
                count += 1
        return count

    def _get_lens(self):
        lens = []
        for f in self.files:
            n_documents = self._get_number_of_documents(f)
            if n_documents is None:
                n_documents = self._get_number_of_documents_by_iteration(f)
            lens.append(n_documents)
        self.lens = lens
        self._len = sum(self.lens)

    def _parse_single_example(self, example):
        data = tf.train.Example.FromString(example)
        data = torch.tensor(list(data.features.feature["text"].int64_list.value), dtype=torch.long)
        if self.mode == "chunks":
            assert data.size(0) == self.seq_len + 1
        return data

    def _process_tfrecord(self, tfrecords_file, resume_idx=None):
        for idx, example in enumerate(tf.io.tf_record_iterator(tfrecords_file)):
            yield self._parse_single_example(example)

    def _maybe_process_tfrecord(self, file_idx):
        if self.processed_files.get(file_idx) is None:
            self.processed_files[file_idx] = list(self._process_tfrecord(self.files[file_idx]))
        return self.processed_files[file_idx]

    def _seek(self, idx):
        cumsum = 0
        for count, (f, length) in cycle(enumerate(zip(self.files, self.lens))):
            prev_cumsum = cumsum
            cumsum += length
            if cumsum == idx:
                remainder = 0
                skip_idx = count + 1
                return skip_idx, remainder
            elif cumsum > idx:
                remainder = idx - prev_cumsum
                skip_idx = count
                return skip_idx, remainder

    def __getitem__(self, idx):
        # seek to correct chunk
        seek_idx, remainder = self._seek(idx)
        f = self.files[seek_idx]
        if self.filetype == "tfrecords":
            chunk = self._maybe_process_tfrecord(
                seek_idx)  # parses tfrecord file to a list *once* then stores in memory
        else:
            raise NotImplementedError
        return chunk[remainder]  # get item from current chunk

    def __len__(self):
        return self._len


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

class TFDSIterDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, batch_size=8, split='train'):
        super().__init__()
        dataset, info = tfds.load(name="ThePile", try_gcs=True, with_info=True)
        self.ds, self.num_examples = dataset[split], info.splits[split].num_examples
        self.data = tfds.as_numpy(self.ds)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        print(f'Items in {split} dataset: ', self.num_examples)
    
    def process_batch(self, batch):
        for ex in batch:
            worker = torch.utils.data.get_worker_info()
            worker_id = id(self) if worker is not None else -1
            example = self.tokenize(ex)
            yield example

    def get_stream(self, examples):
        return chain.from_iterable(map(self.process_batch, cycle(examples)))

    def get_streams(self):
        return zip(*[self.get_stream(self.data)
                    for _ in range(self.batch_size)])

    def __iter__(self):
        return self.get_streams()

    def __len__(self):
        return len(self.examples)
    
    def tokenize(self, item):
        return self.tokenizer.encode(str(item['text'], 'utf-8'), max_length=self.seq_len, truncation=True, padding='max_length', return_tensors='pt')



class TFDSDataset(Dataset):
    def __init__(self, tokenizer, seq_len, split='train'):
        super().__init__()
        dataset, info = tfds.load(name="ThePile", try_gcs=True, with_info=True)
        self.ds, self.num_examples = dataset[split], info.splits[split].num_examples
        self.data = iter(tfds.as_numpy(self.ds))
        self.tokenizer = tokenizer
        #self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        #self.tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
        self.seq_len = seq_len
        print(f'Items in {split} dataset: ', self.num_examples)
    
    def tokenize(self, item):
        #txt = str(item['text'])
        #print(txt)
        return self.tokenizer.encode(str(item['text'], 'utf-8'), max_length=self.seq_len, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, idx):
        return self.tokenize(next(self.data))

    def __len__(self):
        return self.num_examples