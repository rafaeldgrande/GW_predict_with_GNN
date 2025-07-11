
import h5py
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
import random
import copy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import json


def split_data_set(dataset, batch_size, split_ratio=0.8):
    
    # Shuffle the dataset
    np.random.shuffle(dataset)
    
    # Compute split indices
    print('Splitting dataset into train, dev and val sets')
    n_total = len(dataset)
    n_train = int(split_ratio * n_total)
    dev_ratio = (1 - split_ratio) / 2
    n_dev = int(dev_ratio * n_total)
    n_val = n_total - n_train - n_dev  # ensures full coverage

    # Split dataset
    train_dataset = dataset[:n_train]
    dev_dataset = dataset[n_train:n_train + n_dev]
    val_dataset = dataset[n_train + n_dev:]
    
    print(f'Train set size: {len(train_dataset)}')
    print(f'Dev set size: {len(dev_dataset)}')
    print(f'Val set size: {len(val_dataset)}')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, dev_loader, val_loader

def load_params_NN(json_file):
    """
    Loads hyperparameters from a JSON file.

    Args:
        json_file (str): Path to the JSON file containing the hyperparameters.

    Returns:
        dict: A dictionary containing the hyperparameters.
    """
    with open(json_file, 'r') as f:
        return json.load(f)

def load_dataset(file_list_data):
    """
    Loads a dataset from an HDF5 file and returns it as a list of Data objects.

    Args:
        file_list_data (str): Path to a file containing the path to the HDF5 file to load.

    Returns:
        tuple: A tuple containing a list of Data objects and the input dimensionality of the dataset.
    """
    dataset = []
    with open(file_list_data, 'r') as f:
        data_file = f.readline().strip()
        print(f'Loading data from file {data_file}')
        dataset.extend(load_gnn_samples_from_h5(data_file))
    print('Data loaded.')
    input_dim = dataset[0].x.shape[1]  # number of orbital projections per atom
    return dataset, input_dim

def set_seed(seed):

    """
    Set random seeds for Python, NumPy, and PyTorch. This sets the seeds for CPU and GPU.

    Args:
        seed (int): The seed to set.

    Note:
        This sets `torch.backends.cudnn.deterministic = True` and
        `torch.backends.cudnn.benchmark = False` to ensure reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on a given data loader and device.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (DataLoader): DataLoader containing the data to evaluate on.
        device (str or torch.device): Device to use.

    Returns:
        tuple: Tuple containing the predictions and the true targets as NumPy arrays.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.view(-1).cpu().numpy())

    return np.array(preds), np.array(targets)

def build_edge_index_with_pbc(positions, lattice, cutoff=3.0, return_angles=False):
    """
    Returns:
        edge_index: (2, N_edges) torch.long
        edge_attr: (N_edges, 1) distance
        If return_angles:
            angle_index: (3, N_triplets) [i, j, k]
            angles: (N_triplets, 1) angle in radians at atom j between vectors j->i and j->k
    """

    Natoms = positions.shape[0]
    atoms = Atoms(positions=positions, cell=lattice, pbc=True)

    i, j, dvec = neighbor_list("ijd", atoms, cutoff, self_interaction=False)

    i = np.asarray(i)
    j = np.asarray(j)
    dvec = np.asarray(dvec)

    if dvec.ndim == 1:
        dvec = np.vstack(dvec)  # force shape to (N_edges, 3)

    # print("dvec.shape =", dvec.shape)
    # print("i.shape =", i.shape)
    # print("j.shape =", j.shape)

    assert i.shape == j.shape == (dvec.shape[0],), \
        f"Inconsistent shapes: i={i.shape}, j={j.shape}, dvec={dvec.shape}"

    edge_index = torch.tensor(np.stack([i, j]), dtype=torch.long)
    edge_dist = torch.norm(torch.tensor(dvec, dtype=torch.float32), dim=1, keepdim=True)

    assert edge_index.shape[1] == edge_dist.shape[0], \
        f"edge_index has {edge_index.shape[1]} edges, but edge_dist has {edge_dist.shape[0]}"

    if not return_angles:
        return edge_index, edge_dist

    # Build angle triplets (i-j-k)
    from collections import defaultdict

    neighbors_of = defaultdict(list)
    vecs = {}

    # collect neighbors and vectors j→i
    for idx, (ii, jj, dv) in enumerate(zip(i, j, dvec)):
        neighbors_of[jj].append(ii)
        vecs[(jj, ii)] = dv  # vector from j to i

    angle_list = []
    triplet_index = []

    for j_atom in neighbors_of:
        neighbors = neighbors_of[j_atom]
        for i_atom in neighbors:
            for k_atom in neighbors:
                if i_atom == k_atom:
                    continue
                v1 = vecs[(j_atom, i_atom)]
                v2 = vecs[(j_atom, k_atom)]
                cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cosine = np.clip(cosine, -1.0, 1.0)  # numerical safety
                angle = np.arccos(cosine)
                angle_list.append([angle])
                triplet_index.append([i_atom, j_atom, k_atom])  # i-j-k: angle at j

    angles = torch.tensor(angle_list, dtype=torch.float32) if angle_list else torch.empty((0, 1), dtype=torch.float32)
    if triplet_index:
        angle_index = torch.tensor(triplet_index, dtype=torch.long).view(-1, 3).transpose(0, 1)
    else:
        angle_index = torch.empty((3, 0), dtype=torch.long)

    return edge_index, edge_dist, angle_index, angles


def load_gnn_samples_from_h5(filename):
    """
    Loads GNN samples from an HDF5 file and returns a list of Data objects.

    The HDF5 file is expected to contain the following datasets:
        - 'atom_orb_projections': (Nk, Nb, Natoms, Norbs) array of atomic orbital projections
        - 'Edft': (Nb, Nk) array of DFT energies
        - 'qp_corrections': (Nb, Nk) array of quasiparticle corrections
        - 'atomic_positions': (Natoms, 3) array of atomic positions
        - 'lattice_vectors': (3, 3) array of lattice vectors
        - 'atomic_species': (Natoms,) array of atomic species as strings

    Returns a list of Data objects, where each Data object contains the node features,
    edge index, edge attributes, angle index, angles, atomic positions, target value,
    and input value for a single sample.
    """
    
    with h5py.File(filename, 'r') as f:
        projections = f['atom_orb_projections'][()]  # (Nk, Nb, Natoms, Norbs)
        Edft = f['Edft'][()]                         # (Nb, Nk)
        qp_corr = f['qp_corrections'][()]            # (Nb, Nk)
        positions = f['atomic_positions'][()]        # (Natoms, 3)
        lattice = f['lattice_vectors'][()]           # (3, 3)
        species = f['atomic_species'][()].astype(str)

    Nk, Nb, Natoms, Norbs = projections.shape
    data_list = []

    # Move edge/angle computation OUTSIDE the loop!
    edge_index, edge_attr, angle_index, angles = build_edge_index_with_pbc(
        positions, lattice, cutoff=3.0, return_angles=True
    )
    
    # print("edge_index shape:", edge_index.shape)
    # print("edge_attr shape:", edge_attr.shape)

    # 🔍 Check consistency once
    assert edge_index.shape[1] == edge_attr.shape[0], \
        f"Mismatch: edge_index has {edge_index.shape[1]}, edge_attr has {edge_attr.shape[0]}"

    # Convert once for reuse
    edge_index = edge_index.clone()
    edge_attr = edge_attr.clone()
    angle_index = angle_index.clone()
    angles = angles.clone()
    pos_tensor = torch.tensor(positions, dtype=torch.float32)

    for ik in range(Nk):
        for ib in range(Nb):
            node_features = projections[ik, ib, :, :]      # (Natoms, Norbs)
            y = qp_corr[ib, ik]                            # scalar target
            edft = Edft[ib, ik]                            # scalar input

            data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                angle_index=angle_index,
                angles=angles,
                pos=pos_tensor,
                y=torch.tensor([y], dtype=torch.float32),
                edft=torch.tensor([edft], dtype=torch.float32),
                species=species
            )            

            data_list.append(data)

    return data_list

def train_model_with_patience(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, patience=20):
    print('Starting training...')
    train_loss_list = []
    train_mae_list = []
    val_loss_list = []
    val_mae_list = []
    
    best_val_mae = float('inf')
    best_epoch = -1
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())  # init

    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=patience_learning_rate, min_lr=1e-6
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_train_mae = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1))
            mae = F.l1_loss(pred, batch.y.view(-1))

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_graphs
            total_train_mae += mae.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_train_mae = total_train_mae / len(train_loader.dataset)
        train_loss_list.append(avg_train_loss)
        train_mae_list.append(avg_train_mae)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        total_val_mae = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss = loss_fn(pred, batch.y.view(-1))
                val_mae = F.l1_loss(pred, batch.y.view(-1))
                total_val_loss += val_loss.item() * batch.num_graphs
                total_val_mae += val_mae.item() * batch.num_graphs

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_mae = total_val_mae / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        val_loss_list.append(avg_val_loss)
        val_mae_list.append(avg_val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d} of {epochs} | Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | patience_counter {patience_counter} | Current LR: {current_lr:.2e}")

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())  # Save best model
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} because patience counter reached {patience_counter}")
            break

    model.load_state_dict(best_model_weights)  # Restore best model weights
    print(f'Finished this training. Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}')

    return train_loss_list, train_mae_list, val_loss_list, val_mae_list, model, best_val_mae, best_epoch


def plot_mae_loss(train_loss, train_mae, val_loss, val_mae):

    """
    Plot training and validation loss and MAE curves.

    Parameters
    ----------
    train_loss : list
        List of training losses at each epoch.
    train_mae : list
        List of training mean absolute errors at each epoch.
    val_loss : list
        List of validation losses at each epoch.
    val_mae : list
        List of validation mean absolute errors at each epoch.
    """
    f, axs = plt.subplots(figsize=(14, 6), ncols=2)

    plt.sca(axs[0])
    plt.plot(train_loss, label='Train', color='blue')
    plt.plot(val_loss, label='Validation', color='orange')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (eV)')
    plt.legend()
    plt.grid(True)


    plt.sca(axs[1])
    plt.plot(train_mae, label='Train', color='blue')
    plt.plot(val_mae, label='Validation', color='orange')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (eV)')
    plt.legend()
    plt.grid(True)
    # y axis in log scale
    
    plt.tight_layout()
    plt.savefig('loss_mae.png')
    plt.close()

def plot_prediction_vs_true_val(y_pred, y_true):
    """
    Plot a scatter plot of the predicted QP corrections vs the true QP corrections.
    
    Parameters
    ----------
    y_pred : array-like
        Predicted QP corrections.
    y_true : array-like
        True QP corrections.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
    plt.xlabel('True QP Correction (eV)')
    plt.ylabel('Predicted QP Correction (eV)')
    plt.title('Predicted vs True QP Corrections')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pred_vs_true_qp.png')
    plt.close()