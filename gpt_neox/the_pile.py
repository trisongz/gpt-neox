
import tensorflow_datasets as tfds
import tensorflow as tf
import io
import zstandard
import jsonlines
import simdjson as json

tfds.disable_progress_bar()
parser = json.Parser()

_CITATION = """
"""
_DESCRIPTION = """
"""
_DATASET_MODES = ["lm"]

_PILE_URL = 'http://eaidata.bmk.sh/data/pile/train/{}.jsonl.zst'
_PILE_SPLITS = 30

_URLS = {
    'the_pile': {
        'train': [_PILE_URL.format(str(i).zfill(2)) for i in range(_PILE_SPLITS)],
        'test': 'http://eaidata.bmk.sh/data/pile/test.jsonl.zst',
        'validation': 'http://eaidata.bmk.sh/data/pile/val.jsonl.zst',
    }
}


_VERSION = tfds.core.Version('1.0.0')
_RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
}

_NAME = 'the_pile'
_FILE_FORMAT = 'jsonlines'

def json_parser(x):
    try:
        line = parser.parse(x).as_dict()
        return line
    except ValueError:
        return x

class PileReader:
    def __init__(self, filenames, para_joiner='\n\n'):
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.para_joiner = para_joiner

    def _read_fn(self, filename):
        with tf.io.gfile.GFile(filename, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream, loads=json_parser)
            for item in reader:
                result = dict()
                if isinstance(item, str):
                    result['text'] = item
                else:
                    text = item['text']
                    if isinstance(text, list):
                        text = self.para_joiner.join(text)
                        result['text'] = text
                yield result
    
    def __iter__(self):
        for filename in self.filenames:
            return self._read_fn(filename)


class ThePileConfig(tfds.core.BuilderConfig):
    def __init__(self, *, mode=None, **kwargs):
        super(ThePileConfig, self).__init__(
            name=mode,
            description="The Pile dataset",
            **kwargs)

class ThePile(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ThePileConfig(version=_VERSION, mode=mode) for mode in _DATASET_MODES
    ]
    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'text': tfds.features.Text()
            }),
            supervised_keys=("text", "text"),
            homepage='https://github.com/EleutherAI/The-Pile',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        dl_manager.verify_ssl = False
        dl_paths = dl_manager.download(_URLS['the_pile'])
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"paths": dl_paths['train']}),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={"paths": dl_paths['validation']}),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"paths": dl_paths['test']}),
        ]

    def _generate_examples(self, paths):
        pipeline = PileReader(paths)
        for x, result in enumerate(pipeline):
            if result:
                idx = f'{x}_the_pile'
                yield idx, {'text': result['text']}