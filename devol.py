from __future__ import print_function


from genome_handler import GenomeHandler
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist, cifar10
from keras.callbacks import EarlyStopping
from datetime import datetime
import random as rand
import csv
from tqdm import trange, tqdm
import sys
import operator

METRIC_OPS = [ operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [ min, max]


class DEvol:

    def __init__(self, genome_handler, data_path="", verbose_fit=True):
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')
        self.bssf = (None, float('inf'), 0.) # model, loss, accuracy
        self.obj = 'max' if self.genome_handler.metric=='accuracy' else 'min'
        self.metric_index = 1 if self.genome_handler.metric is 'loss' else -1
        self.metric_op = METRIC_OPS[self.obj is 'max']
        self.metric_objective = METRIC_OBJECTIVES[self.obj is 'max']
        self.verbose_fit=verbose_fit

        print("Genome encoding and accuracy data stored at", self.datafile, "\n")





    # Create a population and evolve
    # Returns best model found in the form of (model, accuracy)
    def run(self, dataset, num_generations, pop_size, epochs, fitness=None):
        generations = trange(num_generations, desc="Generations")
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        # Generate initial random population
        members = [self.genome_handler.generate() for _ in range(pop_size)]
        fit = []
        for i in trange(len(members), desc="Gen 1 Models Fitness Eval"):
            res = self.evaluate(members[i], epochs)
            v = res[self.metric_index]
            if self.metric_op(v, self.bssf[self.metric_index]):
                    self.bssf = res
            fit.append(v)
        fit = np.array(fit)
        pop = Population(members, fit, fitness, obj=self.obj)

        tqdm.write("Generation 1:\t\tbest: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}\t\tbest_stats: {3}".format(self.metric_objective(fit), 
            np.mean(fit), np.std(fit), self.bssf[1:]))
        # Evolve over generations
        for gen in generations:
            if gen == 0:
                continue
            members = []
            for i in range(int(pop_size*0.95)): # Crossover
                members.append(self.crossover(pop.select(), pop.select()))
            members += pop.getBest(pop_size - int(pop_size*0.95))
            for i in range(len(members)): # Mutation
                members[i] = self.mutate(members[i], gen)
            fit = []
            for i in trange(len(members), desc="Gen %i Models Fitness Eval" % (gen + 1)):
                res = self.evaluate(members[i], epochs)
                v = res[self.metric_index]
                if self.metric_op(v, self.bssf[self.metric_index]):
                        self.bssf = res
                fit.append(v)
            fit = np.array(fit)
            pop = Population(members, fit, fitness, obj=self.obj)

            tqdm.write("Generation {3}:\t\tbest: {0:0.4f}\t\taverage: {1:0.4f}\t\tstd: {2:0.4f}\t\tbest_stats: {4}".format(self.metric_objective(fit), 
                np.mean(fit), np.std(fit), gen + 1, self.bssf[1:]))
        return self.bssf


    def evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
            epochs=epochs,
            verbose=self.verbose_fit,
            callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=0)])
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        # Record the stats
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss, accuracy]
            writer.writerow(row)  
        return model, loss, accuracy
    
    def crossover(self, genome1, genome2):
        crossIndexA = rand.randint(0, len(genome1))
        child = genome1[:crossIndexA] + genome2[crossIndexA:]
        return child
    
    def mutate(self, genome, generation):
        num_mutations = max(3, generation / 4) # increase mutations as program continues
        return self.genome_handler.mutate(genome, num_mutations)

class Population:

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        # fitnesses -= fitnesses.min()
        # fitnesses /= fitnesses.max()
        self.obj = obj
        scores = fitnesses- fitnesses.min()
        scores /= scores.max()
        if self.obj is 'min':
            scores = 1 - scores

        self.scores = scores
        self.s_fit = sum(self.scores)

    def score(self, fitness):
        return (fitness * 100)**4

    def getBest(self, n):
        combined = [(self.members[i], self.scores[i]) \
                for i in range(len(self.members))]
        sorted(combined, key=(lambda x: x[1]), reverse=True)
        return [x[0] for x in combined[:n]]

    def select(self):
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits > dart:
                return self.members[i]


