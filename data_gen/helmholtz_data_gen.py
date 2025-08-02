import os
import numpy as np
import datasets

"""Hugging Face Datasets loader for Helmholtz samples.

The raw `.npy` file is normalized with fixed dataset statistics before being
returned as `U`, `A`, and the two-channel `UA` tensor.
"""

h_data_max = 0.022490255461219907
h_data_min = -0.02734944077477767
h_data_std = 0.004270045990050305
h_data_mean = 9.46510862536118e-06
h_data_a_max = 2.024150613827505
h_data_a_min = -1.8881286627715816
h_data_a_std = 0.28432797574133245
h_data_a_mean = -2.5465973392311507e-06
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
                "U": datasets.Array3D(dtype="float32", shape=(1,128, 128)),
                "A": datasets.Array3D(dtype="float32", shape=(1,128, 128)),
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
                    data[:, :,0] = (data[:, :,0]-h_data_mean)/(h_data_std+1e-8)
                    data[:, :,1] = (data[:, :,1]-h_data_a_mean)/(h_data_a_std+1e-8)
                    yield fname, {
                        "U": data[...,:1].transpose(2,0,1).astype(np.float32).copy(),
                        "A": data[...,1:].transpose(2,0,1).astype(np.float32).copy(),
                        "UA": data.transpose(2,0,1).astype(np.float32).copy(),
                        
                    }
                except Exception as e:
                    print(f"Failed to load {fname}: {e}")
                    continue
