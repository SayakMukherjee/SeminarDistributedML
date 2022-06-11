from typing import Tuple, Any

import numpy as np
import sklearn
import time
import torch

from fltk.core.node import Node
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer
from fltk.util.config import FedLearningConfig

# Group 10 >> starts
import copy

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
# Group 10 << ends

class Client(Node):
    """
    Federated experiment client.
    """
    running = False

    def __init__(self, identifier: str, rank: int, world_size: int, config: FedLearningConfig):
        super().__init__(identifier, rank, world_size, config)

        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.net.parameters(),
                                                              **self.config.optimizer_args)
        self.scheduler = MinCapableStepLR(self.optimizer,
                                          self.config.scheduler_step_size,
                                          self.config.scheduler_gamma,
                                          self.config.min_lr)

    def remote_registration(self):
        """
        Function to perform registration to the remote. Currently, this will connect to the Federator Client. Future
        version can provide functionality to register to an arbitrary Node, including other Clients.
        @return: None.
        @rtype: None
        """
        self.logger.info('Sending registration')
        self.message('federator', 'ping', 'new_sender')
        self.message('federator', 'register_client', self.id, self.rank)
        self.running = True
        self._event_loop()

    def stop_client(self):
        """
        Function to stop client after training. This allows remote clients to stop the client within a specific
        timeframe.
        @return: None
        @rtype: None
        """
        self.logger.info('Got call to stop event loop')
        self.running = False

    def _event_loop(self):
        self.logger.info('Starting event loop')
        while self.running:
            time.sleep(0.1)
        self.logger.info('Exiting node')

    def train(self, num_epochs: int, round_id: int):
        """
        Function implementing federated learning training loop, allowing to run for a configurable number of epochs
        on a local dataset. Note that only the last statistics of a run are sent to the caller (i.e. Federator).
        @param num_epochs: Number of epochs to run during a communication round's training loop.
        @type num_epochs: int
        @param round_id: Global communication round ID to be used during training.
        @type round_id: int
        @return: Final running loss statistic and acquired parameters of the locally trained network. NOTE that
        intermediate information is only logged to the STD-out.
        @rtype: Tuple[float, Dict[str, torch.Tensor]]
        """
        start_time = time.time()
        running_loss = 0.0
        final_running_loss = 0.0
        for local_epoch in range(num_epochs):
            effective_epoch = round_id * num_epochs + local_epoch
            progress = f'[RD-{round_id}][LE-{local_epoch}][EE-{effective_epoch}]'
            if self.distributed:
                # In case a client occurs within (num_epochs) communication rounds as this would cause
                # an order or data to re-occur during training.
                self.dataset.train_sampler.set_epoch(effective_epoch)

            training_cardinality = len(self.dataset.get_train_loader())
            self.logger.info(f'{progress}{self.id}: Number of training samples: {training_cardinality}')

            for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # Mark logging update step
                if i % self.config.log_interval == 0:
                    self.logger.info(
                            f'[{self.id}] [{local_epoch}/{num_epochs:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                    final_running_loss = running_loss / self.config.log_interval
                    running_loss = 0.0
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f'{progress} Train duration is {duration} seconds')

        return final_running_loss, self.get_nn_parameters()

    def set_tau_eff(self, total):
        client_weight = self.get_client_datasize() / total
        n = self.get_client_datasize()  # pylint: disable=invalid-name
        E = self.config.epochs  # pylint: disable=invalid-name
        B = 16  # nicely hardcoded :) # pylint: disable=invalid-name
        tau_eff = int(E * n / B) * client_weight
        if hasattr(self.optimizer, 'set_tau_eff'):
            self.optimizer.set_tau_eff(tau_eff)

    def test(self) -> Tuple[float, float, np.array]:
        """
        Function implementing federated learning test loop.
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
        @rtype: Tuple[float, float, np.array]
        """
        self.logger.info("Entered test in client")
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        self.logger.info('For loop in test completed')
        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def get_client_datasize(self):  # pylint: disable=missing-function-docstring
        return len(self.dataset.get_train_sampler())

    def exec_round(self, num_epochs: int, round_id: int) -> Tuple[Any, Any, Any, Any, float, float, float, np.array]:
        """
        Function as access point for the Federator Node to kick off a remote learning round on a client.
        @param num_epochs: Number of epochs to run
        @type num_epochs: int
        @return: Tuple containing the statistics of the training round; loss, weights, accuracy, test_loss, make-span,
        training make-span, testing make-span, and confusion matrix.
        @rtype: Tuple[Any, Any, Any, Any, float, float, float, np.array]
        """
        self.logger.info(f"[EXEC] running {num_epochs} epochs locally...")
        start = time.time()
        self.logger.info('Calling train for exec_round')
        loss, weights = self.train(num_epochs, round_id)
        time_mark_between = time.time()
        self.logger.info('Calling test from exec_round')
        accuracy, test_loss, test_conf_matrix = self.test()

        end = time.time()
        round_duration = end - start
        train_duration = time_mark_between - start
        test_duration = end - time_mark_between
        self.logger.info("Done with client training for this round")
        #self.logger.info(f'Round duration is {duration} seconds')

        if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
            self.optimizer.pre_communicate()
        for k, value in weights.items():
            weights[k] = value.cpu()
        return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, test_conf_matrix

    # Group 10 >> starts
    def get_stats(self) -> Tuple[Any, Any, Any]:
        self.logger.info('Entered get_stats in client')

        activations = {}
        model = copy.deepcopy(self.net)
        model.fc = Identity()
        self.logger.info(f'[ID:{self.id}] {model}')

        start_time = time.time()

        self.logger.info(f'[ID:{self.id}] Starting feature extraction')

        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)

                try:
                    activations['outputs'].data = torch.cat((activations['outputs'], outputs.clone()), 0)
                    activations['labels'].data = torch.cat((activations['labels'], labels.clone()), 0)
                except:
                    activations['outputs'] = outputs.clone()
                    activations['labels'] = labels.clone()

        self.logger.info(f'[ID:{self.id}] Feature extraction completed')

        means_dict = {}
        std_dict = {}
        sizes_dict = {}

        for class_name in activations['labels'].unique():
            means_dict[class_name.item()] = activations['outputs'][activations['labels'] == class_name].mean(0)
            std_dict[class_name.item()] = activations['outputs'][activations['labels'] == class_name].std(0)
            sizes_dict[class_name.item()] = torch.sum(activations['labels'] == class_name)

            self.logger.info(f'[ID:{self.id}] Means dimension {means_dict[class_name.item()].shape}')

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'[ID:{self.id}] Data stats fetched in {duration} seconds')

        return means_dict, std_dict, sizes_dict
    # Group 10 << ends

    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')
