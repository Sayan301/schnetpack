'''
    Default implementation of SchNet performance on MD17 dataset.

'''

import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
from schnetpack.datasets import ethanol
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
        num_train=20,
        num_val=1,
        num_test=1,
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
    return ethanol_data

def printDataInfo(ethanoldata):

    # print("For ethanol dataset")
    # print(ethanoldata.dataset[0]['energy'])
    # print("For ethanol test dataset")
    # print(ethanoldata.train_dataset[0]['energy'])
    # print(type(ethanoldata))
    # print(type(ethanoldata.dataset))
    # print(type(ethanoldata.train_dataset))
    # for data in ethanoldata.train_dataset:
    #     print("Energy = ", data['energy'])
    properties = ethanoldata.train_dataset[0]
    print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
    # atomrefs = ethanoldata.train_dataset.atomrefs
    # print('U0 of hyrogen:', atomrefs[ethanol.U0][1].item(), 'eV')
    # print('U0 of carbon:', atomrefs[ethanol.U0][6].item(), 'eV')
    # print('U0 of oxygen:', atomrefs[ethanol.U0][8].item(), 'eV')
    # print('U0 of nitrogen:', atomrefs[ethanol.U0][7].item(), 'eV')
    # print('U0 of fluorine:', atomrefs[ethanol.U0][9].item(), 'eV')
    # means, stddevs = ethanoldata.get_stats(
    #     ethanol.U0, divide_by_atoms=True, remove_atomref=True
    # )
    # print('Mean atomization energy / atom:', means.item())
    # print('Std. dev. atomization energy / atom:', stddevs.item())


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
    
def training(ethanoldata, outputlabels):
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
    # pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=ethanol.U0)
    pred_atoms = spk.atomistic.atomwise.AtomClassifier(n_in=n_atom_basis, n_out=len(outputlabels) , output_key='scalar_representation')

    #Setting up the Neural Network Potential
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_atoms],
        # output_modules=[pred_energy],
        pretrain = True,
        # postprocessors=[
        #     trn.CastTo64(),
        #     trn.AddOffsets(ethanol.U0, add_mean=True, add_atomrefs=True)
        # ]
    )
    device = torch.device("cuda")
    model_path = os.path.join(ethanoltut, "best_inference_model")
    model = torch.load(model_path, map_location=device)

    #Setting up the output specifications
    output_U0 = spk.task.ModelOutput(
        name='scalar_representation',
        loss_fn=torch.nn.CrossEntropyLoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    #Setup the training model
    task = spk.task.AtomisticPretrain(
        model=model,
        outputs=[output_U0],
        # outputs=[output_energy],
        pretrain_labels = outputlabels,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4}
    )


    logger = pl.loggers.TensorBoardLogger(save_dir=ethanoltut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(ethanoltut, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss"
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=ethanoltut,
        max_epochs=100, # for testing, we restrict the number of epochs
    )

    #Training
    trainer.fit(task, datamodule=ethanoldata)

def onehot_encodings(batch_labels, outputlabels):
        labels = torch.tensor([outputlabels[str(label)] for label in batch_labels])
        return labels

def inference(ethanoltut, ethanol_data, outputlabels):
        
    device = torch.device("cuda")
    # load model
    model_path = os.path.join(ethanoltut, "best_inference_model")
    best_model = torch.load(model_path, map_location=device)

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
    )
    accurate_pred_count_list = []
    for structure in ethanol_data.test_dataset:
    # convert atoms to SchNetPack inputs and perform prediction
        atoms = Atoms(numbers=structure[spk.properties.Z], positions=structure[spk.properties.R])
        inputs = converter(atoms)
        # print('Loaded properties:\n', *['{:s}\n'.format(i) for i in inputs.keys()])
        
        result = torch.argmax(best_model(inputs)['scalar_representation'], dim = 1)
        truth = onehot_encodings(structure['_atomic_numbers'].tolist(), outputlabels).to('cuda')
        
        accurate_pred_count_list.append((len(truth) - torch.count_nonzero(result - truth))/len(truth))
    
    print("Accuracy = ", sum(accurate_pred_count_list)/len(accurate_pred_count_list))

if __name__=='__main__':
    pretrain_outputlabels = {'1':0, '6':1, '7':2, '8':3, '9':4}
    #Location for the tensorboard logger and best_inference_model 
    ethanoltut = './checking'
    if not os.path.exists(ethanoltut):
        os.makedirs(ethanoltut)

    #Partition file for train-val-test split
    filename = "/home/sayan/Documents/schnetmod/schnetpack/split.npz"
    if os.path.exists(filename):
        os.remove(filename)
        print("Removed split.npz file")

    ethanoldata = getDataset(ethanoltut)
    printDataInfo(ethanoldata)

    print("-----TRAINING PHASE-----")
    start = datetime.datetime.now()
    training(ethanoldata, pretrain_outputlabels)
    end = datetime.datetime.now()

    print("-----INFERENCE PHASE-----")
    ethanoltut = './testing'
    if not os.path.exists(ethanoltut):
        os.makedirs(ethanoltut)

    #Partition file for train-val-test split
    test_filename = "/home/sayan/Documents/schnetmod/schnetpack/split2.npz"
    if os.path.exists(test_filename):
        os.remove(test_filename)
        print("Removed split2.npz file")

    ethanoldata = get_testData(ethanoltut)
    inference(ethanoltut, ethanoldata, pretrain_outputlabels)
    print("Output for pretraining 1 with 1000 points and 500 epochs")
    print("Total time = ", end - start)
    
