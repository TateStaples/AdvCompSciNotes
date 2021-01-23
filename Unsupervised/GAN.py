from random import *


# todo: check all of this works

class GeneralAdversarialNetwork:
    latent_dimensions = 100

    def __init__(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator
        self.gan = self.generator.extend(self.discriminator)
        self.data = None

    def generate_latent_space(self, n_samples):
        latent_space = [[gauss(mu=0, sigma=5) for i in range(self.latent_dimensions)] for j in range(n_samples)]
        return latent_space

    def train(self, dataset, n_epochs, n_batches):
        self.data = dataset
        data_per_epoch = len(dataset[0]) // n_batches
        half_batch = n_batches // 2
        for e in range(n_epochs):
            for i in range(data_per_epoch):
                real_data, real_labels = self.get_real_examples(half_batch)
                fake_data, fake_labels = self.get_fake_examples(half_batch)
                data, labels = real_data.extend(fake_data), real_labels.extend(fake_labels)
                self.discriminator.train(data, labels)
                latent_space = self.generate_latent_space(data_per_epoch)
                labels = [1 for i in range(data_per_epoch)]
                self.update(latent_space, labels)

    def update(self, data, labels):
        combined = self.generator.extend(self.discriminator)
        combined.train(data, labels)
        len_of_gen = len(self.generator.network)
        self.generator.network = combined.network[:len_of_gen]

    def evaluate(self, test_amount):
        real_data, real_labels = self.get_real_examples(test_amount//2)
        fake_data, fake_labels = self.get_fake_examples(test_amount//2)
        real_accuracy = self.discriminator.evaluate(real_data, real_labels)
        fake_accuracy = self.discriminator.evaluate(fake_data, fake_labels)
        print(f"Real accuracy: {real_accuracy}%, Fake accuracy: {fake_accuracy}%")

    def get_real_examples(self, amount):
        data, labels = self.data
        chosen_ints = [randint(0, len(data)) for i in range(amount)]
        return [data[i] for i in chosen_ints], [labels[i] for i in chosen_ints]

    def get_fake_examples(self, amount):
        latent_stuff = self.generate_latent_space(amount)
        generated = [self.generator.forward_propogate(rand) for rand in latent_stuff]
        labels = [0 for i in range(amount)]  # the labels are 0 because they are fake
        return generated, labels


