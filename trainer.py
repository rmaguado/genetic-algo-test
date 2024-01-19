import numpy as np
import json
import sys

from pong import Pong, SettingsPong
from brain import Brain, SettingsBrain


class SettingsTrainer:
    def __init__(self, config):
        self.max_generations = config["max_generations"]
        self.population_size = config["population_size"]
        self.death_rate = config["death_rate"]
        self.steps_per_game = config["steps_per_game"]
        self.debug = config["debug"]
        self.save_path = config["save_path"]
        self.load_path = config["load_path"]


class Trainer:
    """
    Evolves by pitching two brains against each other and selecting the best.
    """

    def __init__(self, settings_trainer, settings_pong, settings_brain):
        self.settings = settings_trainer
        self.population = [Brain(settings_brain)
                           for _ in range(self.settings.population_size)]
        self.pong = Pong(settings_pong)
        self.generation = 0
        self.fitness = np.zeros(self.settings.population_size)
        self.num_weak = int(self.settings.population_size *
                            self.settings.death_rate)
        self.correlation = 0

        self.save_path = self.settings.save_path

    def _get_fitness(self):
        self.fitness = np.zeros(self.settings.population_size)
        for i, brain in enumerate(self.population):
            self.fitness[i] += self._evaluate_brain(brain)

    def _evaluate_brain(self, brain):
        score = 0
        self.pong.reset()
        for _ in range(self.settings.steps_per_game):
            action = brain.predict(self.pong.get_state())
            score += self.pong.step(action)
            if self.settings.debug:
                self.pong.render()
                input()

        return score

    def play_indefinitely(self, brain):
        score = 0
        self.pong.reset()
        while True:
            action = brain.predict(self.pong.get_state())
            score += self.pong.step(action)
            self.pong.render()
            print(f"Score: {score}")
            close = input()
            if close == "q":
                break

    def _select(self):
        self._get_fitness()

        selected_fitness = self.fitness[:self.settings.population_size - self.num_weak]

        self.correlation = -np.corrcoef(
            range(len(selected_fitness)), selected_fitness)[0, 1]

        self.population = [self.population[i]
                           for i in np.argsort(self.fitness)[::-1]]

        for i in range(self.num_weak):
            self.population[i].replicate(self.population[-1-i])

        for i in range(self.num_weak, self.settings.population_size):
            self.population[i].mutate()

    def _generation_statistics(self):
        print(f"Generation {self.generation}")
        print(
            f"Max score: {np.max(self.fitness)} Min score: {np.min(self.fitness)}")
        print(f"Average score: {np.mean(self.fitness)}")
        print(f"R: {self.correlation}")
        print(f"Top 3: {[ind.id for ind in self.population[:3]]}")
        print()

    def run(self):
        while self.generation < self.settings.max_generations:
            self._select()
            self._generation_statistics()
            self.generation += 1
        best_brain = self.population[-1]
        if self.settings.save_path is not False:
            best_brain.save(self.settings.save_path)
        self.play_indefinitely(best_brain)


def play():
    with open("settings.json", encoding="utf-8") as f:
        settings = json.load(f)
    pong = Pong(SettingsPong(settings["pong"]))
    brain = Brain(SettingsBrain(settings["brain"]))
    if settings["trainer"]["load_path"] is not False:
        brain.load(settings["trainer"]["load_path"])
    while True:
        action = brain.predict(pong.get_state())
        pong.step(action)
        pong.render()
        close = input()
        if close == "q":
            break


def train():
    with open("settings.json", encoding="utf-8") as f:
        settings = json.load(f)
    trainer = Trainer(
        SettingsTrainer(settings["trainer"]),
        SettingsPong(settings["pong"]),
        SettingsBrain(settings["brain"])
    )
    trainer.run()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "play":
            play()
        else:
            print("Invalid argument. Please specify 'train' or 'play'.")
    else:
        print("No argument provided. Please specify 'train' or 'play'.")
