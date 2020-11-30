import pickle
from glob import glob
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset


class NETTACKDataset(InMemoryDataset):
    def __init__(
        self, root=Path("/u/scratch2/eutiushe/CS590MLG-project/nettack_dataset_gen")
    ):
        super(NETTACKDataset, self).__init__(None, self.transform_data)

        root = Path(root)
        self.surrogate_params = pickle.load(
            open(root / "surrogate_params.pickle", "rb")
        )

        # original values on an unmodified graph
        self.og_x = torch.tensor(self.surrogate_params["_X_obs"].todense())

        A = self.surrogate_params["_A_obs"]
        A = A.tocoo()
        row = torch.from_numpy(A.row).to(torch.long)
        col = torch.from_numpy(A.col).to(torch.long)
        self.og_edge_index = torch.stack([row, col], dim=0)

        self.og_y = self.surrogate_params["_z_obs"]

        self.data_paths = sorted(glob(str(root / "dataset" / "*.pickle")))
        self.data, self.slices = self.collate(
            list(map(self.load_data_object, self.data_paths))
        )

    def transform_data(self, data):
        data.x = self.og_x.clone()
        data.edge_index = self.og_edge_index.clone()

        for feat_pertb, structure_pertb in zip(
            data.feature_perturbations, data.structure_perturbations
        ):
            if feat_pertb != ():
                data.x[feat_pertb] = 1 - data.x[feat_pertb]
            elif structure_pertb != ():
                u, v = structure_pertb
                e1_idx = (
                    (data.edge_index.T == torch.tensor((u, v)))
                    .all(1)
                    .nonzero(as_tuple=True)[0]
                )
                e2_idx = (
                    (data.edge_index.T == torch.tensor((v, u)))
                    .all(1)
                    .nonzero(as_tuple=True)[0]
                )

                if e1_idx.shape == (0,) and e2_idx.shape == (0,):
                    # we need to add the edges
                    data.edge_index = torch.cat(
                        (
                            torch.tensor((u, v)).reshape(2, 1),
                            torch.tensor((v, u)).reshape(2, 1),
                            data.edge_index,
                        ),
                        1,
                    )
                else:
                    # we need to remove the edges
                    i = min(e1_idx[0], e2_idx[0])
                    j = max(e1_idx[0], e2_idx[0])

                    data.edge_index = torch.cat(
                        (
                            data.edge_index[:, :i],
                            data.edge_index[:, i + 1 : j],
                            data.edge_index[:, j+1:],
                        ),
                        1,
                    )

        return data

    def load_data_object(self, data_path):
        obj = pickle.load(open(data_path, "rb"))
        data = Data(
            y=self.og_y,
            attacked_node=obj["params"]["u"],
            structure_perturbations=obj["structure_perturbations"],
            feature_perturbations=obj["feature_perturbations"],
            direct_attack=obj["params"]["direct_attack"],
        )
        return data
