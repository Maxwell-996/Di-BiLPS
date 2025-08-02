import os
import numpy as np
import datasets

"""Hugging Face Datasets loader for Poisson samples.

The loader applies the fixed scaling used by the downstream VAE and diffusion
training scripts before emitting the two-channel `UA` tensor.
"""

_CITATION = ""
_DESCRIPTION = "A dataset of .npy files containing U and A matrices."
_HOMEPAGE = ""

class MyMatrixDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=datasets.Version("1.0.0"))
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "UA": datasets.Array3D(dtype="float32", shape=(2,128, 128))
            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": self.config.data_dir},
            )
        ]

    def _generate_examples(self, data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".npy"):
                path = os.path.join(data_dir, fname)
                try:
                    data = np.load(path, allow_pickle=True)
                    
                    UA = data.transpose(2,0,1).astype(np.float32).copy()
                    UA[0,...] = UA[0,...]*36.5
                    UA[1,...] = UA[1,...]/2.15
                    yield fname, {
                        "UA": UA,
                    }
                except Exception as e:
                    print(f"Failed to load {fname}: {e}")
                    continue
