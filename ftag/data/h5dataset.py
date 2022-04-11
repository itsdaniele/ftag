from re import S
import h5py
import json

import torch
from torch.utils import data

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, batch_size=32, num_samples=int(1e6)):
        super().__init__()
        hf = h5py.File(file_path, "r")

        # group_key_x = list(hf.keys())[0]
        # group_key_y = list(hf.keys())[1]

        group_key_x = "X_tracks_loose_train"
        group_key_y = "Y_train"
        group_key_global = "X_train"

        self.ds_x = hf[group_key_x][:num_samples]
        self.ds_global = hf[group_key_global][:num_samples, :2][:, None, :]
        self.ds_global = np.repeat(self.ds_global, repeats=self.ds_x.shape[1], axis=1)
        self.ds_x = np.concatenate((self.ds_x, self.ds_global), axis=-1)

        self.ds_y = hf[group_key_y][:num_samples]

        self.batch_size = batch_size

    def __getitem__(self, index):

        return (
            torch.from_numpy(
                self.ds_x[
                    index * self.batch_size : (index * self.batch_size)
                    + self.batch_size
                ]
            ).float(),
            torch.from_numpy(
                self.ds_y[
                    index * self.batch_size : (index * self.batch_size)
                    + self.batch_size
                ]
            ),
        )

    def __len__(self):
        # return self.ds_x.__len__() // self.batch_size
        return 5000


def get_track_mask(tracks: np.ndarray) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    tracks : np.ndarray
        Loaded tracks with shape (nJets, nTrks, nTrkFeatures). Note, the
        input tracks should not already be converted with np.nan_to_num, as this
        function relies on a np.isnan check in the case where the valid flag is
        not present.

    Returns
    -------
    np.ndarray
        A bool array (nJets, nTrks), True for tracks that are present.

    Raises
    ------
    ValueError
        If no 'valid' flag or at least one float variable in your input tracks.
    """

    # try to use the valid flag, present in newer samples
    if "valid" in tracks.dtype.names:
        return tracks["valid"]

    # instead look for a float variable to use, which will be NaN
    # for the padded tracks
    for var, dtype in tracks.dtype.fields.items():
        if "f" in dtype[0].str:
            return ~np.isnan(tracks[var])

    raise ValueError(
        "Need 'valid' flag or at least one float variable in your input tracks."
    )


class HDF5DatasetTest(data.Dataset):
    def __init__(self, file_path, scale_dict_path, batch_size=32):
        super().__init__()

        classes_to_remove = [15]

        with open(scale_dict_path) as json_file:
            scale_dict = json.load(json_file)["tracks_loose"]

        # variables = list(scale_dict.keys())
        variables = [
            "d0",
            "z0SinTheta",
            "dphi",
            "deta",
            "qOverP",
            "IP3D_signed_d0_significance",
            "IP3D_signed_z0_significance",
            "phiUncertainty",
            "thetaUncertainty",
            "qOverPUncertainty",
            "numberOfPixelHits",
            "numberOfSCTHits",
            "numberOfInnermostPixelLayerHits",
            "numberOfNextToInnermostPixelLayerHits",
            "numberOfInnermostPixelLayerSharedHits",
            "numberOfInnermostPixelLayerSplitHits",
            "numberOfPixelSharedHits",
            "numberOfPixelSplitHits",
            "numberOfSCTSharedHits",
            "numberOfPixelHoles",
            "numberOfSCTHoles",
        ]

        hf = h5py.File(file_path, "r")
        tracks = hf["tracks_loose"][: int(1e6)]

        track_mask = get_track_mask(tracks)

        var_arr_list = []
        for var in variables:
            x = tracks[var]
            shift = np.float32(scale_dict[var]["shift"])
            scale = np.float32(scale_dict[var]["scale"])
            x = np.where(track_mask, x - shift, 0)
            x = np.where(track_mask, x / scale, 0)

            var_arr_list.append(x)

        x = np.stack(var_arr_list, axis=-1)
        y = hf["jets"]["HadronConeExclTruthLabelID"][: int(1e6)]

        indices_toremove = []
        for class_id in classes_to_remove:
            indices_toremove.append(np.where(y == class_id)[0])

        for elem in indices_toremove:
            self.x = np.delete(x, elem, axis=0)
            y = np.delete(y, elem, axis=0)

        y = np.where(y == 4, 1, y)
        y = np.where(y == 5, 2, y)

        enc = OneHotEncoder()
        self.y = enc.fit_transform(y[:, None]).todense()
        self.batch_size = batch_size

    def __getitem__(self, index):

        return (
            torch.from_numpy(
                np.array(
                    self.x[
                        index * self.batch_size : (index * self.batch_size)
                        + self.batch_size
                    ]
                )
            ).float(),
            torch.from_numpy(
                self.y[
                    index * self.batch_size : (index * self.batch_size)
                    + self.batch_size
                ]
            ),
        )

    def __len__(self):
        # return self.ds_x.__len__() // self.batch_size
        return self.x.__len__() // self.batch_size
