#!/usr/bin/env python
import os, sys, math
import numpy as np
import random
import click
import glob
import torch
import espaloma as esp
import dgl
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# GLOBAL PARAMETER
HARTEE_TO_KCALPERMOL = 627.5
RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


#-------------------------
# SUBMODULE
#-------------------------
def _load_datasets(datasets, input_prefix):
    """
    Load unique molecules (nonisomeric smile).
    """
    logging.debug(f"# LOAD UNIQUE MOLECULES")
    for i, dataset in enumerate(datasets):
        path = os.path.join(input_prefix, dataset)
        
        # RNA-nucleoside handled as training set since it only contains 4 entries.
        if dataset == "rna-nucleoside":
            _ds_tr = esp.data.dataset.GraphDataset.load(path)
        else:
            ds = esp.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
            _ds_tr, _ds_vl, _ds_te = ds.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])

        # Merge datasets
        if i == 0:
            ds_tr = _ds_tr
        else:
            ds_tr += _ds_tr
        logging.debug(f"{dataset}: {len(_ds_tr)} entries (total: {len(ds_tr)})")
    del _ds_tr, _ds_vl, _ds_te

    return ds_tr


def _load_duplicate_datasets(ds_tr, input_prefix):
    """
    Load duplicated molecules (isomeric smiles) from different datasets 
    to avoid overlapping molecules in train, validate, test dataset.
    """
    entries = glob.glob(os.path.join(input_prefix, "duplicated-isomeric-smiles-merge", "*"))
    random.seed(RANDOM_SEED)
    random.shuffle(entries)

    n_entries = len(entries)
    entries_tr = entries[:int(n_entries*TRAIN_RATIO)]
    entries_vl = entries[int(n_entries*TRAIN_RATIO):int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO)]
    entries_te = entries[int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO):]
    logging.debug(f"Found {n_entries} entries. Split data into {len(entries_tr)}:{len(entries_vl)}:{len(entries_te)} entries.")
    assert n_entries == len(entries_tr) + len(entries_vl) + len(entries_te)

    for entry in entries_tr:
        _datasets = os.listdir(entry)
        for _dataset in _datasets:
            _ds_tr = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
            ds_tr += _ds_tr
            del _ds_tr

    return ds_tr


def _fn(g):
    """
    Remove unnecessary data from graph.
    """
    g.nodes['g'].data.pop('u_qm')
    g.nodes['g'].data.pop('u_gaff-1.81')
    g.nodes['g'].data.pop('u_gaff-2.11')
    g.nodes['g'].data.pop('u_openff-1.2.0')
    g.nodes['g'].data.pop('u_openff-2.0.0')
    g.nodes['n1'].data.pop('u_qm_prime')
    g.nodes['n1'].data.pop('u_gaff-1.81_prime')
    g.nodes['n1'].data.pop('u_gaff-2.11_prime')
    g.nodes['n1'].data.pop('u_openff-1.2.0_prime')
    g.nodes['n1'].data.pop('u_openff-2.0.0_prime')
    try:
        g.nodes['g'].data.pop('u_amber14')
        g.nodes['n1'].data.pop('u_amber14_prime')
    except:
        pass
    # Remove u_ref_relative. u_ref_relative will be recalculated after handling heterographs with different size
    g.nodes['g'].data.pop('u_ref_relative')          
    g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'].double()
    g.nodes['n1'].data['q_ref'] = g.nodes['n1'].data['q_ref'].float()
    return g


def _augment_conformations(ds_tr, n_max_confs):
    """
    Augment conformations to handle heterographs.

    This is a work around to handle different graph size (shape). DGL requires at least one dimension with same size. 
    Here, we will modify the graphs so that each graph has the same number of conformations instead fo concatenating 
    graphs into heterogenous graphs with the same number of conformations. This will allow batching and shuffling 
    during the training. 
    """
    _ds_tr = []
    for i, g in enumerate(ds_tr):
        n = g.nodes['n1'].data['xyz'].shape[1]
        #logging.debug(f">{i}: {n} conformations")

        if n == n_max_confs:
            # Calculate u_ref_relative
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref'].detach().clone()
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'] - g.nodes['g'].data['u_ref_relative'].mean(dim=-1, keepdims=True)
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'].float()
            g.nodes['g'].data.pop('u_ref')
            _ds_tr.append(g.heterograph)

        elif n < n_max_confs:
            random.seed(RANDOM_SEED)
            index = random.choices(range(0, n), k=n_max_confs-n)            
            #logging.debug(f"Randomly select {len(index)} conformers")

            import copy
            _g = copy.deepcopy(g)

            #print(index)
            a = torch.cat((_g.nodes['g'].data['u_ref'], _g.nodes['g'].data['u_ref'][:, index]), dim=-1)
            b = torch.cat((_g.nodes['n1'].data['xyz'], _g.nodes['n1'].data['xyz'][:, index, :]), dim=1)
            c = torch.cat((_g.nodes['n1'].data['u_ref_prime'], _g.nodes['n1'].data['u_ref_prime'][:, index, :]), dim=1)

            # Update in place
            _g.nodes["g"].data["u_ref"] = a
            _g.nodes["n1"].data["xyz"] = b
            _g.nodes['n1'].data['u_ref_prime'] = c

            # Calculate u_ref_relative
            _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref'].detach().clone()
            _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref_relative'] - _g.nodes['g'].data['u_ref_relative'].mean(dim=-1, keepdims=True)
            _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref_relative'].float()
            _g.nodes['g'].data.pop('u_ref')
            _ds_tr.append(_g.heterograph)

        else:
            random.seed(RANDOM_SEED)
            idx_range = random.sample(range(n), k=n)
            for j in range(n // n_max_confs + 1):
                import copy
                _g = copy.deepcopy(g)

                if (j+1)*n_max_confs > n:
                    _index = range(j*n_max_confs, n)
                    random.seed(RANDOM_SEED)
                    index = random.choices(range(0, n), k=(j+1)*n_max_confs-n)
                    #logging.debug(f"Iteration {j}: Randomly select {len(index)} conformers")

                    a = torch.cat((_g.nodes['g'].data['u_ref'][:, index], _g.nodes['g'].data['u_ref'][:, _index]), dim=-1)
                    b = torch.cat((_g.nodes['n1'].data['xyz'][:, index, :], _g.nodes['n1'].data['xyz'][:, _index, :]), dim=1)
                    c = torch.cat((_g.nodes['n1'].data['u_ref_prime'][:, index, :], _g.nodes['n1'].data['u_ref_prime'][:, _index, :]), dim=1)       
                else:            
                    idx1 = j*n_max_confs
                    idx2 = (j+1)*n_max_confs
                    _index = idx_range[idx1:idx2]
                    #logging.debug(f"Iteration {j}: Extract indice from {idx1} to {idx2}")

                    a = _g.nodes['g'].data['u_ref'][:, _index]
                    b = _g.nodes['n1'].data['xyz'][:, _index, :]
                    c = _g.nodes['n1'].data['u_ref_prime'][:, _index, :]

                # Update in place
                _g.nodes["g"].data["u_ref"] = a
                _g.nodes["n1"].data["xyz"] = b
                _g.nodes["n1"].data["u_ref_prime"] = c

                # Calculate u_ref_relative
                _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref'].detach().clone()
                _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref_relative'] - _g.nodes['g'].data['u_ref_relative'].mean(dim=-1, keepdims=True)
                _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref_relative'].float()
                _g.nodes['g'].data.pop('u_ref')
                _ds_tr.append(_g.heterograph)

    return _ds_tr


#-------------------------
# MAIN
#-------------------------
def run(kwargs):
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    layer = kwargs['layer']
    units = kwargs['units']
    config = kwargs['config']
    janossy_config = kwargs['janossy_config']
    learning_rate = kwargs['learning_rate']
    output_prefix = kwargs['output_prefix']
    input_prefix = kwargs['input_prefix']
    datasets = kwargs['datasets']
    n_max_confs = kwargs['n_max_confs']
    force_weight = kwargs['force_weight']

    # Convert config and janossy_config into list
    _config = []
    for _ in config.split():
        try:
            _config.append(int(_))
        except:
            _config.append(str(_))
    config = _config

    _janossy_config = []
    for _ in janossy_config.split():
        try:
            _janossy_config.append(int(_))
        except:
            _janossy_config.append(str(_))
    janossy_config = _janossy_config

    # Convert datasets into list
    _datasets = [ str(_) for _ in datasets.split() ]
    datasets = _datasets

    # Load datasets
    logging.debug(f"# LOAD DUPLICATED MOLECULES")
    ds_tr = _load_datasets(datasets, input_prefix)
    logging.debug(f"# Training size is now: {len(ds_tr)}")
    
    # Load duplicate datasets
    ds_tr = _load_duplicate_datasets(ds_tr, input_prefix)
    logging.debug(f"# Training size is now: {len(ds_tr)}")

    # Remove unnecessary data from graph
    from espaloma.graphs.utils.regenerate_impropers import regenerate_impropers
    ds_tr.apply(_fn, in_place=True)
    ds_tr.apply(regenerate_impropers, in_place=True)

    # Handle heterographs
    logging.debug(f"# AUGMENT CONFORMATIONS TO HANDLE HETEROGRAPHS")
    ds_tr_augment = _augment_conformations(ds_tr, n_max_confs)
    logging.debug(f"# Training size is now: {len(ds_tr_augment)}")
    del ds_tr

               
    #
    # Define espaloma model
    #

    # Representation
    #layer = esp.nn.layers.dgl_legacy.gn(layer)
    layer = esp.nn.layers.dgl_legacy.gn(layer, {"aggregator_type": "mean", "feat_drop": 0.1})
    representation = esp.nn.Sequential(layer, config=config)

    # out_features: Define modular MM parameters Espaloma will assign
    # 1: atom hardness and electronegativity
    # 2: bond linear combination, enforce positive
    # 3: angle linear combination, enforce positive
    # 4: torsion barrier heights (can be positive or negative)
    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=units, config=janossy_config,
        out_features={
                1: {'s': 1, 'e': 1},
                2: {'log_coefficients': 2},
                3: {'log_coefficients': 2},
                4: {'k': 6},
        },
    )
    readout_improper = esp.nn.readout.janossy.JanossyPoolingWithSmirnoffImproper(in_features=units, config=janossy_config, out_features={"k": 2})

    class ExpCoeff(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
            g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
            return g

    class GetLoss(torch.nn.Module):
        def energy_loss(self, g):
            return torch.nn.MSELoss()(
                g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True),
                g.nodes['g'].data['u_ref_relative'],
            )
        def charge_loss(self, g):
            return torch.nn.MSELoss()(
                g.nodes['n1'].data['q'],
                g.nodes['n1'].data['q_ref'],
            )
        def force_loss(self, g):
            du_dx_hat = torch.autograd.grad(
                g.nodes['g'].data['u'].sum(),
                g.nodes['n1'].data['xyz'],
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            du_dx = g.nodes["n1"].data["u_ref_prime"]

            return torch.nn.MSELoss()(
                du_dx, 
                du_dx_hat
            )
        def forward(self, g):
            #logging.debug(f"Loss_charge:      {self.charge_loss(g)}")
            #logging.debug(f"Loss_energy:      {self.energy_loss(g)}")
            #logging.debug(f"Loss_force:       {self.force_loss(g)}")
            loss = self.charge_loss(g) + self.energy_loss(g) + self.force_loss(g) * force_weight
            if g.number_of_nodes('n4_improper') > 0:
                #logging.debug(f"Loss_n4_improper: {g.nodes['n4_improper'].data['k'].pow(2).mean()}")
                loss = loss + g.nodes['n4_improper'].data['k'].pow(2).mean()
            if g.number_of_nodes('n4') > 0:
                #logging.debug(f"Loss_n4:          {g.nodes['n4'].data['k'].pow(2).mean()}")
                loss = loss + g.nodes['n4'].data['k'].pow(2).mean()
            return loss
        
    net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
            ExpCoeff(),
            esp.nn.readout.charge_equilibrium.ChargeEquilibrium(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
            GetLoss(),
    ).cuda()

    # Check if checkpoint file exists
    checkpoints = glob.glob("{}/*.th".format(output_prefix))
    if checkpoints:
        n = [ int(c.split('net')[1].split('.')[0]) for c in checkpoints ]
        n.sort()
        last_step = n[-1]
        last_checkpoint = os.path.join(output_prefix, "net{}.th".format(last_step))
        net.load_state_dict(torch.load(last_checkpoint))
        step = last_step + 1
        print('Found checkpoint file ({}). Restrating from step {}'.format(last_checkpoint, step))
    else:
        step = 1

    # Train
    ds_tr_loader = dgl.dataloading.GraphDataLoader(ds_tr_augment, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    with torch.autograd.set_detect_anomaly(True):
        for idx in range(step, step+epochs):
            for g in ds_tr_loader:
                optimizer.zero_grad()
                g = g.to("cuda:0")
                g.nodes["n1"].data["xyz"].requires_grad = True 
                loss = net(g)
                loss.backward()
                optimizer.step()
            if idx % 10 == 0:
                # Note: returned loss is a joint loss of different units.
                print(idx, HARTEE_TO_KCALPERMOL * loss.pow(0.5).item())
                if not os.path.exists(output_prefix):
                    os.mkdir(output_prefix)
                torch.save(net.state_dict(), output_prefix + "/net%s.th" % idx)


@click.command()
@click.option("-e",   "--epochs",          default=1, help="number of epochs", type=int)
@click.option("-b",   "--batch_size",      default=128, help="batch size", type=int)
@click.option("-l",   "--layer",           default="SAGEConv", type=click.Choice(["SAGEConv", "GATConv", "TAGConv", "GINConv", "GraphConv"]), help="GNN architecture")
@click.option("-u",   "--units",           default=128, help="GNN layer", type=int)
@click.option("-act", "--activation",      default="relu", type=click.Choice(["relu", "leaky_relu"]), help="activation method")
@click.option("-c",   "--config",          default="128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions)", type=str)
@click.option("-jc",   "--janossy_config", default="128 relu 128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions)", type=str)
@click.option("-lr",  "--learning_rate",   default=1e-5, help="learning rate", type=float)
@click.option("-i",   "--input_prefix",    default="data", help="input prefix to graph data", type=str)
@click.option("-d",   "--datasets",        help="name of the dataset", type=str)
@click.option("-o",   "--output_prefix",   default="output", help="output prefix to save checkpoint network models", type=str)
@click.option("-n",   "--n_max_confs",     default="50", help="number of conformations to reshape the graph", type=int)
@click.option("-w",   "--force_weight",    default=1.0, type=float)
def cli(**kwargs):
    logging.debug(kwargs)
    logging.debug(esp.__version__)
    run(kwargs)


if __name__ == "__main__":
    cli()
