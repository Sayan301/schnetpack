'''
    Default implementation of SchNet performance on MD17 dataset.

'''

import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
from schnetpack.datasets import QM9
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
from schnetpack.datasets import MD17
from ase import Atoms


def getDataset(qm9tut):
    '''
    This function is used to extract a portion of the main database with some specifications.

    INPUTS:
        qm9tut = Location of the tensorflow model logger

    PRINTS: Different properties for the dataset

    RETURNS: The extracted dataset portion

    '''
    #Obtaining the dataset to be used
    qm9data = QM9(
        os.path.join(qm9tut,'qm9.db'),
        batch_size=10,
        num_train=100,
        num_val=2,
        num_test=2,
        train_transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
            trn.CastTo32()
        ],
        val_transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(QM9.U0, remove_mean=True, remove_atomrefs=True),
            trn.CastTo32()
        ],
        test_transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(QM9.U0, remove_mean=False, remove_atomrefs=True),
            trn.CastTo32()
        ],
        property_units={QM9.U0: 'eV'},
        num_workers=1,
        split_file=os.path.join(qm9tut, "split.npz"),
        pin_memory=True, # set to false, when not using a GPU
        load_properties=[QM9.U0], #only load U0 property
    )
    qm9data.prepare_data()
    qm9data.setup()
    print("Training size = ", len(qm9data.train_dataset))
    return qm9data

def printDataInfo(qm9data):

    # print("For ethanol dataset")
    # print(qm9data.dataset[0]['energy'])
    # print("For ethanol test dataset")
    # print(qm9data.train_dataset[0]['energy'])
    # print(type(qm9data))
    # print(type(qm9data.dataset))
    # print(type(qm9data.train_dataset))
    # for data in qm9data.train_dataset:
    #     print("Energy = ", data['energy'])
    properties = qm9data.train_dataset[0]
    print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
    atomrefs = qm9data.train_dataset.atomrefs
    print('U0 of hyrogen:', atomrefs[QM9.U0][1].item(), 'eV')
    print('U0 of carbon:', atomrefs[QM9.U0][6].item(), 'eV')
    print('U0 of oxygen:', atomrefs[QM9.U0][8].item(), 'eV')
    print('U0 of nitrogen:', atomrefs[QM9.U0][7].item(), 'eV')
    print('U0 of fluorine:', atomrefs[QM9.U0][9].item(), 'eV')
    means, stddevs = qm9data.get_stats(
        QM9.U0, divide_by_atoms=True, remove_atomref=True
    )
    print('Mean atomization energy / atom:', means.item())
    print('Std. dev. atomization energy / atom:', stddevs.item())

def training(qm9data):
    '''
    Training of the model
    '''
    cutoff = 5.
    n_atom_basis = 30

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    #Properties to be predicted
    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)

    #Setting up the Neural Network Potential
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_U0],
        # output_modules=[pred_energy],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)
        ]
    )

    #Setting up the output specifications
    output_U0 = spk.task.ModelOutput(
        name=QM9.U0,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    #Setup the training model
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_U0],
        # outputs=[output_energy],

        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(qm9tut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=qm9tut,
        max_epochs=3, # for testing, we restrict the number of epochs
    )

    #Training
    trainer.fit(task, datamodule=qm9data)

def inference(qm9tut, qm9data):
        
    device = torch.device("cpu")
    # load model
    model_path = os.path.join(qm9tut, "best_inference_model")
    best_model = torch.load(model_path, map_location=device)
    # best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'))

    # # set up converter
    # converter = spk.interfaces.AtomsConverter(
    #     neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    # )
    results = []
    for batch in qm9data.test_dataloader():
        # print(batch)
        result = best_model(batch)
        results.append(result['energy_U0'])
    print("Result = ",results)
    # print("Size of result = ", result.shape)
    # create atoms object from dataset
    mse = nn.MSELoss()
    mse_energy = []; mse_forces = []

    # for structure in qm9data.test_dataset:
    # # convert atoms to SchNetPack inputs and perform prediction
    #     atoms = Atoms(numbers=structure[spk.properties.Z], positions=structure[spk.properties.R])
    #     inputs = converter(atoms)
    #     # print('Loaded properties:\n', *['{:s}\n'.format(i) for i in inputs.keys()])

    #     results = best_model(inputs)
    #     res_energy = results['energy'].cpu()
    #     res_forces = results['forces'].cpu()

    #     # print("Predicted energy = ", res_energy)
    #     # print("Predicted forces = ", res_forces)

    #     true_energy = structure['energy']
    #     true_forces = structure['forces']
    #     # print("True energy = ", true_energy)
    #     # print("True forces = ", true_forces)

    #     mse_energy.append(mse(res_energy, true_energy))
    #     mse_forces.append(mse(res_forces, true_forces))
        
    # print("Test data size = ", len(mse_energy))
    # print("Avg. Energy MSE = ", sum(mse_energy)/len(mse_energy))
    # print("Avg. Forces MSE = ", sum(mse_forces)/len(mse_forces))


if __name__=='__main__':
    #Location for the tensorboard logger and best_inference_model 
    qm9tut = './qm9'
    if not os.path.exists(qm9tut):
        os.makedirs(qm9tut)

    #Partition file for train-val-test split
    filename = "/home/sayan/Documents/schnetmod/schnetpack/split.npz"
    if os.path.exists(filename):
        os.remove(filename)
        print("Removed split.npz file")

    qm9data = getDataset(qm9tut)
    printDataInfo(qm9data)
    print("-----TRAINING PHASE-----")
    start = datetime.datetime.now()
    training(qm9data)
    end = datetime.datetime.now()

    print("-----INFERENCE PHASE-----")
    inference(qm9tut, qm9data)
    print("Output for qm9dataset")
    print("Total time = ", end - start)

    
