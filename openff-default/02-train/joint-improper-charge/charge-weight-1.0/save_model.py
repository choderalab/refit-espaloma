#!/usr/bin/env python
import os, sys, math
import numpy as np
import random
import click
import glob
import torch
import espaloma as esp
import dgl


def save_model(kwargs):
    best_model = kwargs['model']

    # settings
    layer="SAGEConv"
    units=512
    activation="relu"
    config=[units, "relu", 0.1, units, "relu", 0.1, units, "relu", 0.1]
    janossy_config=[units, "relu", 0.1, units, "relu", 0.1, units, "relu", 0.1, units, "relu", 0.1]

    # representations
    #layer = esp.nn.layers.dgl_legacy.gn(layer)
    layer = esp.nn.layers.dgl_legacy.gn(layer, {"aggregator_type": "mean", "feat_drop": 0.1})
    representation = esp.nn.Sequential(layer, config=config)
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
        
    # network model
    net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
            esp.nn.readout.janossy.ExpCoefficients(),
            esp.nn.readout.charge_equilibrium.ChargeEquilibrium(),
            esp.nn.readout.janossy.LinearMixtureToOriginal(),
    )

    print("saving {}".format(best_model))
    state_dict = torch.load(best_model, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
    model_name = best_model.split('/')[-1].split('.')[0]
    torch.save(net, model_name + ".pt")


@click.command()
@click.option('--model', required=True, help='name of the network model')
def cli(**kwargs):
    save_model(kwargs)


if __name__ == "__main__":
    cli()