import os
import numpy as np
import datasets

"""Hugging Face Datasets loader for bounded Navier-Stokes samples.

Bounded NS data includes an additional `cond` vector `(cx, cy, r)` that is
consumed by the conditional CLIP encoder.
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
                "UA": datasets.Array3D(dtype="float32", shape=(2,128, 128)),
                "cond": datasets.Array2D(dtype="float32", shape=(1,3))
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
        num_files = len(os.listdir(data_dir))
        for id in range(num_files//2):
            fname = os.path.join(data_dir,f"merge_{id}.npy")
            bfname = os.path.join(data_dir,f"bound_msg_{id}.npy")

            try:
                data = np.load(fname, allow_pickle=True)
                bdmsg = np.load(bfname, allow_pickle=True)
                U = data[...,8].astype(np.float32).copy()
                A =  data[...,4].astype(np.float32).copy()
                UA = np.stack([U,A], axis = 0)
                cx,cy,r = bdmsg[0],bdmsg[1],bdmsg[2]
                cond = np.array([cx,cy,r]).reshape(1,3)
                yield fname, {
                    "UA": UA,
                    "cond": cond
                }
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
                continue
