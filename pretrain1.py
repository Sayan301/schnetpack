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
        batch_size=100,
        num_train=1000,
        num_val=100,
        num_test=100,
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

def get_testData(ethanoltut):
    ethanol_data = MD17(
        os.path.join(ethanoltut,'ethanol.db'),
        molecule='ethanol',
        batch_size=10,
        num_train=20,
        num_val=1,
        num_test=100,
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
    print("Test size = ", len(ethanol_data.train_dataset))
    return ethanol_data

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
    # atomrefs = qm9data.train_dataset.atomrefs
    # print('U0 of hyrogen:', atomrefs[QM9.U0][1].item(), 'eV')
    # print('U0 of carbon:', atomrefs[QM9.U0][6].item(), 'eV')
    # print('U0 of oxygen:', atomrefs[QM9.U0][8].item(), 'eV')
    # print('U0 of nitrogen:', atomrefs[QM9.U0][7].item(), 'eV')
    # print('U0 of fluorine:', atomrefs[QM9.U0][9].item(), 'eV')
    # means, stddevs = qm9data.get_stats(
    #     QM9.U0, divide_by_atoms=True, remove_atomref=True
    # )
    # print('Mean atomization energy / atom:', means.item())
    # print('Std. dev. atomization energy / atom:', stddevs.item())

def training(qm9data, outputlabels):
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
    # pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=QM9.U0)
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
        #     trn.AddOffsets(QM9.U0, add_mean=True, add_atomrefs=True)
        # ]
    )

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
        model=nnpot,
        outputs=[output_U0],
        # outputs=[output_energy],
        pretrain_labels = outputlabels,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4},
        logfile = '/home/sayan/Documents/schnetmod/schnetpack/atom_pretrain.csv'
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

def onehot_encodings(batch_labels, outputlabels):
        labels = torch.tensor([outputlabels[str(label)] for label in batch_labels])
        return labels

def inference(qm9tut, ethanol_data, outputlabels):
        
    device = torch.device("cuda")
    # load model
    model_path = os.path.join(qm9tut, "best_inference_model")
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
    qm9tut = './checking'
    if not os.path.exists(qm9tut):
        os.makedirs(qm9tut)

    #Partition file for train-val-test split
    filename = "/home/sayan/Documents/schnetmod/schnetpack/split.npz"
    if os.path.exists(filename):
        os.remove(filename)
        print("Removed split.npz file")

    logfile = '/home/sayan/Documents/schnetmod/schnetpack/atom_pretrain.csv'
    if os.path.exists(logfile):
        os.remove(logfile)

    qm9data = getDataset(qm9tut)
    printDataInfo(qm9data)

    print("-----TRAINING PHASE-----")
    start = datetime.datetime.now()
    training(qm9data, pretrain_outputlabels)
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
    inference(qm9tut, ethanoldata, pretrain_outputlabels)
    print("Output for pretraining 1 with 1000 points and 500 epochs")
    print("Total time = ", end - start)
    
