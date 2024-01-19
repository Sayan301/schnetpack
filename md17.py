'''
    Default implementation of SchNet performance on MD17 dataset.

'''

import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
from schnetpack.datasets import MD17
from ase import Atoms


def getDataset(forcetut):
    '''
    This function is used to extract a portion of the main database with some specifications.

    INPUTS:
        forcetut = Location of the tensorflow model logger

    PRINTS: Different properties for the dataset

    RETURNS: The extracted dataset portion

    '''
    #Obtaining the dataset to be used
    ethanol_data = MD17(
        os.path.join(forcetut,'ethanol.db'),
        molecule='ethanol',
        batch_size=10,
        num_train=2000,
        num_val=2,
        num_test=2,
        train_transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ],
        val_transforms=[
            trn.ASENeighborList(cutoff=5.),
            trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
            trn.CastTo32()
        ],
        num_workers=20,
        pin_memory=True, # set to false, when not using a GPU
    )
    ethanol_data.prepare_data()
    ethanol_data.setup()
    print("Training size = ", len(ethanol_data.train_dataset))
    print("Validation size = ", len(ethanol_data.val_dataset))
    print("Test size = ", len(ethanol_data.test_dataset))
    return ethanol_data

def printDataInfo(ethanol_data):

    # print("For ethanol dataset")
    # print(ethanol_data.dataset[0]['energy'])
    # print("For ethanol test dataset")
    # print(ethanol_data.train_dataset[0]['energy'])
    # print(type(ethanol_data))
    # print(type(ethanol_data.dataset))
    # print(type(ethanol_data.train_dataset))
    # for data in ethanol_data.train_dataset:
    #     print("Energy = ", data['energy'])
    properties = ethanol_data.train_dataset[0]
    print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
    print("Atomic numbers = ", properties['_atomic_numbers'])


def training(ethanol_data):
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
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=MD17.energy)
    pred_forces = spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces)

    #Setting up the Neural Network Potential
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        # pretrain = False,
        # output_modules=[pred_energy],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(MD17.energy, add_mean=True, add_atomrefs=False)
        ]
    )
    #Setting up the output specifications
    output_energy = spk.task.ModelOutput(
        name=MD17.energy,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.01,
        # loss_weight=1,

        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name=MD17.forces,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=0.99,
        # loss_weight=0,

        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    #Setup the training model
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        # outputs=[output_energy],
        logfile='/home/sayan/Documents/schnetmod/schnetpack/schnet.csv',
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3}
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(forcetut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=forcetut,
        max_epochs=3, # for testing, we restrict the number of epochs
    )

    #Training
    trainer.fit(task, datamodule=ethanol_data)

def inference(forcetut, ethanol_data):
        
    device = torch.device("cuda")
    # load model
    model_path = os.path.join(forcetut, "best_inference_model")
    best_model = torch.load(model_path, map_location=device)
    # print("Best model pretrain = ",best_model.pretrain)
    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )

    
    # create atoms object from dataset
    mse = nn.MSELoss()
    mse_energy = []; mse_forces = []

    for structure in ethanol_data.test_dataset:
    # convert atoms to SchNetPack inputs and perform prediction
        atoms = Atoms(numbers=structure[spk.properties.Z], positions=structure[spk.properties.R])
        inputs = converter(atoms)
        # print('Loaded properties:\n', *['{:s}\n'.format(i) for i in inputs.keys()])

        results = best_model(inputs)
        res_energy = results['energy'].cpu()
        res_forces = results['forces'].cpu()

        print("Predicted energy = ", res_energy)
        # print("Predicted forces = ", res_forces)

        true_energy = structure['energy']
        true_forces = structure['forces']
        print("True energy = ", true_energy)
        # print("True forces = ", true_forces)

        mse_energy.append(mse(res_energy, true_energy))
        mse_forces.append(mse(res_forces, true_forces))
        break
        
    print("Test data size = ", len(mse_energy))
    print("Avg. Energy MSE = ", sum(mse_energy)/len(mse_energy))
    print("Avg. Forces MSE = ", sum(mse_forces)/len(mse_forces))


if __name__=='__main__':
    #Location for the tensorboard logger and best_inference_model 
    forcetut = './md17'
    if not os.path.exists(forcetut):
        os.makedirs(forcetut)

    #Partition file for train-val-test split
    filename = "/home/sayan/Documents/schnetmod/schnetpack/split.npz"
    if os.path.exists(filename):
        os.remove(filename)
        print("Removed split.npz file")

    logfile = '/home/sayan/Documents/schnetmod/schnetpack/schnet.csv'
    if os.path.exists(logfile):
        os.remove(logfile)
        
    ethanol_data = getDataset(forcetut)
    # printDataInfo(ethanol_data)
    print("-----TRAINING PHASE-----")
    start = datetime.datetime.now()
    training(ethanol_data)
    end = datetime.datetime.now()

    print("-----INFERENCE PHASE-----")
    inference(forcetut, ethanol_data)
    print("Output for 10,000 samples of ethanol with force regularizer")
    print("Total time = ", end - start)

    
