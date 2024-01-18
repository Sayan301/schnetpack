'''
    Default implementation of SchNet performance on MD17 dataset.

'''

import torch
import copy
import random
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
from schnetpack.datasets import MD17
from ase import Atoms
import numpy as np
from schnetpack.data import ASEAtomsData


def getDataset(log_folder, batch_size, train_size, val_size, test_size):
    '''
    This function is used to extract a portion of the main database with some specifications.

    INPUTS:
        log_folder = Location of the tensorflow model logger

    PRINTS: Different properties for the dataset

    RETURNS: The extracted dataset portion

    '''
    #Obtaining the dataset to be used
    ethanol_data = MD17(
        os.path.join(log_folder,'ethanol.db'),
        molecule='ethanol',
        batch_size=batch_size,
        num_train=train_size,
        num_val=val_size,
        num_test=test_size,
        # train_transforms=[
        #     trn.ASENeighborList(cutoff=5.),
        #     trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
        #     trn.CastTo32()
        # ],
        # val_transforms=[
        #     trn.ASENeighborList(cutoff=5.),
        #     trn.RemoveOffsets(MD17.energy, remove_mean=True, remove_atomrefs=False),
        #     trn.CastTo32()
        # ],
        num_workers=20,
        pin_memory=True, # set to false, when not using a GPU
    )
    ethanol_data.prepare_data()
    ethanol_data.setup()
    print("Original training size : ", len(ethanol_data.train_dataset))
    return ethanol_data

def printDataInfo(ethanol_data):
    '''
    This function displays the relevant informations regarding the dataset and train-val-test split

    INPUTs :
        ethanol_data = Concerned dataset
    '''
    print("For ethanol dataset")
    print("For ethanol test dataset")
    print(ethanol_data.train_dataset[0]['energy'])

    properties = ethanol_data.test_dataset[0]
    print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
    # print(type(properties))
    # for keys in properties:
    #     print(keys)
    #     print('Properties', type(properties[keys]))
    #     print('Properties', properties[keys]))


def add_noise(max_noise = 0.01):
    '''
    This function returns the amount of noise N(0,1) to be added to the position of an atom.

    INPUTS:
        max_noise = variance 
    OUTPUTS:
        noise
    '''
    return np.random.normal(loc = 0, scale = 0.01)


def perturb(positions, index_to_perturb):
    '''
    Given an atomic structure, this function randomly chooses one (or more) atoms and purturbs their position.

        INPUTS:
            positions = position vectors of each atom
            index_to_perturb = the atom whose position is to be perturbed

        OUTPUTS:
            new (perturbed) positions of the atoms
    '''
    for index in index_to_perturb:
        for i in range(3):
            positions[index][i] = positions[index][i] + add_noise()
    return positions

def taylor_approx_energy(energy, forces, pos, new_pos):
    '''
    This function computes the energy of a perturbed structure using first order Taylor series expansion on the original structure.

    INPUTS:
        energy = energy of the original structure
        force = force components of the original structure
        pos = position vectors of the original structure
        new_pos = position vectors of the perturbed structure

    OUTPUTS:
        energy of the perturbed structure
    '''
    change_in_energy = np.trace(np.matmul(forces,(new_pos-pos).T))
    new_energy = energy + change_in_energy
    return new_energy 


def data_augmentation(log_folder, ethanol_data, batch_size, train_size, val_size, test_size, aug_scale, n_indices):
    '''
    Given a dataset, this function obtains an augmented dataset by randomly perturbing the position vector of the atoms and 
    calculating the energy using first order Taylor series expansions.

    INPUTS:
        log_folder = file location of original dataset
        ethanol_data = original dataset

    OUTPUTS:
        augmented dataset
    '''
    partition_size = int(len(ethanol_data.train_dataset)/aug_scale) + val_size
    
    atomic_numbers = ethanol_data.train_dataset[0]['_atomic_numbers'].numpy()
    atoms_list = []
    property_list = []
    # print(atomic_numbers.shape[0])
    for ind in range(partition_size):
    
        positions = ethanol_data.train_dataset[ind]['_positions']
        energy = ethanol_data.train_dataset[ind]['energy']
        forces = ethanol_data.train_dataset[ind]['forces']

        ats = Atoms(positions = positions, numbers = atomic_numbers)
        properties = {'energy': energy.numpy(), 'forces': forces.numpy()}
        property_list.append(properties)
        atoms_list.append(ats)

        index_to_perturb = random.sample(range(0, atomic_numbers.shape[0] - 1), n_indices)
        # for ele in forces:
        #     force_magnitudes = np.linalg.norm(ele)
        # force_deceasing_indices = np.array(force_magnitudes).argsort()[::-1]    
        # index_to_perturb = force_deceasing_indices[:n_indices]

        for count in range(aug_scale - 1): 

            #First perturbed structure
            old_positions = positions.clone()
            new_positions = perturb(positions, index_to_perturb)
            new_energy = taylor_approx_energy(energy, forces, old_positions, new_positions)
            new_ats = Atoms(positions = new_positions, numbers = atomic_numbers)
            new_properties = {'energy': new_energy.numpy(), 'forces': forces.numpy()}
            property_list.append(properties)
            atoms_list.append(ats)

        # #Second perturbed structure
        # old_positions = positions.clone()
        # new_positions = perturb(positions, atomic_numbers.shape[0])
        # new_energy = taylor_approx_energy(energy, forces, old_positions, new_positions)
        # new_ats = Atoms(positions = new_positions, numbers = atomic_numbers)
        # new_properties = {'energy': new_energy.numpy(), 'forces': forces.numpy()}
        # property_list.append(properties)
        # atoms_list.append(ats)

    print("Total structures = ", len(atoms_list))
    filename = os.path.join(log_folder,'ethanol_augmented.db')
    if os.path.exists(filename):
        os.remove(filename)

    new_dataset = ASEAtomsData.create(
        os.path.join(log_folder,'ethanol_augmented.db'),
        distance_unit = 'Ang',
        property_unit_dict = {'energy':'kcal/mol', 'forces':'kcal/mol/Ang'}
    )
    
    new_dataset.add_systems(property_list, atoms_list)
    ethanol_augdata = spk.data.AtomsDataModule(
        os.path.join(log_folder,'ethanol_augmented.db'), 
        batch_size=batch_size,
        num_train=train_size,
        num_val=val_size,
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
    ethanol_augdata.prepare_data()
    ethanol_augdata.setup()
    print("Augmented training size : ", len(ethanol_augdata.train_dataset))
    return ethanol_augdata

def training(log_folder, ethanol_data, alpha, epochs):
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
        loss_weight=alpha,
        # loss_weight=1,

        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name=MD17.forces,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1 - alpha,
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

        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-3}
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=log_folder)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(log_folder, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=log_folder,
        max_epochs=epochs, # for testing, we restrict the number of epochs
    )

    #Training
    trainer.fit(task, datamodule=ethanol_data)


def inference(log_folder, ethanol_data):
        
    device = torch.device("cuda")
    # load model
    model_path = os.path.join(log_folder, "best_inference_model")
    best_model = torch.load(model_path, map_location=device)

    # set up converter
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )

    
    # create atoms object from dataset
    mse = nn.MSELoss()
    mse_energy = []; mse_forces = []
    mae = nn.L1Loss()
    mae_energy = []; mae_forces = []
    print("Test data sample")
    print(ethanol_data.test_dataset[0]['energy'])
    # structure = ethanol_data.test_dataset[0]
    for structure in ethanol_data.test_dataset:
        # convert atoms to SchNetPack inputs and perform prediction
        atoms = Atoms(numbers=structure[spk.properties.Z], positions=structure[spk.properties.R])
        inputs = converter(atoms)
        # print('Loaded properties:\n', *['{:s}\n'.format(i) for i in inputs.keys()])

        results = best_model(inputs)
        res_energy = results['energy'].cpu()
        res_forces = results['forces'].cpu()
        # print("Predicted energy = ", res_energy)
        # print("Predicted forces = ", res_forces)

        true_energy = structure['energy']
        true_forces = structure['forces']
        # print("True energy = ", true_energy)
        # print("True forces = ", true_forces)
        mse_energy.append(mse(res_energy, true_energy))
        mse_forces.append(mse(res_forces, true_forces))

        mae_energy.append(mae(res_energy, true_energy))
        mae_forces.append(mae(res_forces, true_forces))
        
    print("Test data size = ", len(mae_energy))
    print("Avg. Energy MSE = ", math.sqrt(sum(mse_energy)/len(mse_energy)))
    print("Avg. Forces MSE = ", math.sqrt(sum(mse_forces)/len(mse_forces)))
    
    print("Avg. Energy MAE = ", sum(mae_energy)/len(mae_energy))
    print("Avg. Forces MAE = ", sum(mae_forces)/len(mae_forces))


if __name__=='__main__':
    #Hyper parameters for training

    batch_size = 100
    train_size = 40000 # (N+1)x augmented = x Training + Nx perturbed
    val_size = 100
    test_size = 100
    alpha = 1
    epochs = 5500
    aug_scale = 4 #Scale of agumentation
    n_indices = 4 #No of atoms to be perturbed

    #Location for the tensorboard logger and best_inference_model 
    log_folder = './three_pert_4x'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    #Partition file for train-val-test split
    filename = "/home/sayan/Documents/schnetpack/split.npz"
    if os.path.exists(filename):
        os.remove(filename)
        print("Removed split.npz file")

    ethanol_data = getDataset(log_folder, batch_size, train_size, val_size, test_size)
    ethanol_augdata = data_augmentation(log_folder, ethanol_data, batch_size, train_size, val_size, test_size, aug_scale, n_indices)

    start = datetime.datetime.now()
    training(log_folder, ethanol_augdata, alpha, epochs)
    end = datetime.datetime.now()

    inference(log_folder, ethanol_data)

    print("Output for 10,000 ethanol samples 3x perturbed and hence 4x augmented")

    print("Batch Size = ", batch_size)
    print("Training size = ", train_size)
    print("Epochs = ", epochs)
    print("Aug scale = ",aug_scale)
    print("No of atoms perturbed = ", n_indices)
    print("Total time = ", end - start)

    
