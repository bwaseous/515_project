import os
import json
from time import time
from copy import copy

from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim

from transformers import (
    RobertaForTokenClassification, 
    RobertaTokenizerFast,
    RobertaConfig,
    MobileViTV2ForImageClassification,
    MobileViTFeatureExtractor,
    MobileViTV2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

import datasets
from datasets import load_dataset, load_metric

from project.utils import create_dir, set_seed
from project.model import CreditDefaultNet

from typing import List, Union, Optional, Dict, Tuple, Any, Literal

class BaseDataset:

    def __init__(
            self
        ) -> None:

        self.dataset_name: str = "BaseDataset"

        self.dataset: datasets.dataset_dict.DatasetDict = None

        self.Train = None
        self.Test = None

        self.model = None
    
    def prepare_dataset(self) -> None:
        """
        Method to load dataset for training model
        Returns Train, Test into self
        """

        return NotImplementedError
    
    def load_model(self) -> None:
        """
        Method to prepare model for training or
        load trained model for inference
        """

        return NotImplementedError
    
    def train(self):
        """
        Method to train/fine-tune model
        """

        return NotImplementedError
    
    def get_model(self):
        """
        Method to retrieve trained model
        """

        return NotImplementedError
    
    def infer(self):
        """
        Method to run inference on fine-tuned model
        """

        return NotImplementedError
    


class CreditDefault(BaseDataset):

    def __init__(
            self,
            model: Optional[nn.Module] = None
        ) -> None:
        super().__init__()

        self.dataset_name: str = "credit-default"

        self.dataset = load_dataset(
            path = "imodels/credit-card",
        )

        model: nn.Module = model


    def prepare_dataset(
            self, 
            device: Literal["cpu", "cuda"]
        ) -> None:
        self.device = device
        
        self.Train = self.dataset_to_tensor(self.dataset["train"].with_format("torch"))
        self.Test = self.dataset_to_tensor(self.dataset["test"].with_format("torch"))

        return None
    
    def dataset_to_tensor(
            self,
            d
        ) -> torch.Tensor:
        
        n = len(d.features)
        m = len(d)
        
        t = torch.zeros((m,n), device = self.device)
        
        for i, feature in enumerate(d.features):
            t[:,i] = d[feature]
            
        return t
    
    def load_model(self, device:Literal["cpu", "cuda"], model: Optional[nn.Module] = CreditDefaultNet) -> None:
        self.model = model().to(device)
        return None

    def train_one_epoch(self, optimizer: optim.Optimizer, batch_size: int, shuffle: Optional[bool] = True) -> Tuple[List[float]]:
        batch_loss = []
        accuracy_list = []

        dataloader = DataLoader(self.Train, batch_size, shuffle = shuffle)
        
        for i, batch in enumerate(dataloader):
            
            optimizer.zero_grad()
            outputs = self.model(batch[:,:-1]).reshape(-1)
            
            loss = self.loss(outputs, batch[:,-1])

            loss.backward()
            optimizer.step()
            accuracy = (outputs==batch[:,-1]).to('cpu').numpy().mean()
            batch_loss.append(loss.item())
            accuracy_list.append(accuracy)
            print(f"\tbatch {i+1} | loss: {loss.item()} | accuracy: {accuracy}")

        return batch_loss, accuracy_list
    
    def test_one_epoch(self, batch_size: int, shuffle: Optional[bool] = True) -> Tuple[List[float]]:
        self.model.eval()
        batch_loss = []
        accuracy_list = []

        with torch.no_grad():
            dataloader = DataLoader(self.Test, batch_size, shuffle = shuffle)
            
            for i, batch in enumerate(dataloader):

                outputs = self.model(batch[:,:-1]).reshape(-1)
                
                loss = self.loss(outputs, batch[:,-1])
                accuracy = (outputs==batch[:,-1]).to('cpu').numpy().mean()
                batch_loss.append(loss.item())
                accuracy_list.append(accuracy)
                print(f"\tbatch {i+1} | loss: {loss.item()} | accuracy: {accuracy}")
                
        return batch_loss, accuracy_list
                

    def train(
            self,
            result_csv: Union[str, Path],
            optimizers: Dict[str, Tuple[optim.Optimizer, Dict[str, Any]]],
            schedulers: Dict[str, Tuple[optim.lr_scheduler.LRScheduler, Dict[str, Any]]],
            loss_func: Optional[nn.modules.loss._Loss] = nn.MSELoss,
            epochs: Optional[int] = 50,
            batch_size: Optional[int] = 20,
            model: Optional[nn.Module] = CreditDefaultNet,
            device: Optional[Literal["cpu", "cuda"]] = "cpu"
        ) -> pd.DataFrame:

        self.device = device
        
        assert((self.Train != None) and (self.Test != None)), "run prepare_dataset method"
        
        self.loss: nn.modules.loss._Loss = loss_func()
        
        result_csv = Path(result_csv)
        if not result_csv.parents[0].exists():
            result_csv.parents[0].makedir(parents = True)
        
        df = pd.DataFrame(
            index = list(optimizers.keys()), 
            columns = list(schedulers.keys())
        )
        df.to_csv(result_csv)

        for scheduler_name, (scheduler, scheduler_params) in schedulers.items():
            for optimizer_name, (optimizer, optimizer_params) in optimizers.items():
                t0 = time()

                self.load_model(model = model, device = device)
                print(f"-------------------------------------")
                print(f"|{scheduler_name}, {optimizer_name}|")
                _optimizer = copy(optimizer)(self.model.parameters(), **optimizer_params)
                _scheduler = copy(scheduler)(_optimizer, **scheduler_params)
                
                best_loss = 1000.

                run_data = {
                    "train_loss": [], 
                    "test_loss": [], 
                    "train_accuracy": [],
                    "test_accuracy": [],
                    "train_time": 0.,
                }

                for epoch in range(epochs):
                    print(f"Epoch {epoch+1}")
                    self.model.train(True)

                    train_loss, train_accuracy = self.train_one_epoch(_optimizer, batch_size)
                    
                    run_data["train_loss"].append(train_loss)
                    run_data["train_accuracy"].append(train_accuracy)

                    test_loss, test_accuracy  = self.test_one_epoch(batch_size)
                    run_data["test_loss"].append(test_loss)
                    run_data["test_accuracy"].append(test_accuracy)
                    
                    print(f"\tAverage accuracy {np.mean(test_accuracy)}")
                    
                    if np.mean(test_loss) < best_loss:
                        best_loss = np.mean(test_loss)
                        model_path = Path(f"models/credit/{optimizer_name}_{scheduler_name}.pt")
                        if not model_path.parents[0].exists():
                            model_path.parents[0].mkdir(parents = True)
                        
                        torch.save(self.model.state_dict(), model_path)

                    if scheduler != optim.lr_scheduler.ReduceLROnPlateau:
                        _scheduler.step()
                    else:
                        _scheduler.step(np.mean(test_loss))
                
                run_data["train_time"] = time() - t0
                df.at[optimizer_name,scheduler_name] = run_data
                df.to_csv(result_csv)

                self.model.to("cpu")
                
                del self.model
                del _optimizer
                del _scheduler
            
        return df
    
    def get_model(self) -> Union[None, nn.Module]:
        if self.model:
            return self.model
        
        else:
            print("No model trained")
            return None
    

    

class PII(BaseDataset):

    def __init__(self):
        super().__init__()

        self.dataset_name: str = "PII"

        self.dataset = load_dataset(
            path = "conll2003"
        )
        self.tokenized_dataset = None

        self.tokenizer = None
        self.model = None
        self.metric = None
        
        self.label2id = dict()
        self.id2label = dict()

    def load_tokenizer(
            self,
            tokenizer_path: Union[str, Path],
            device: Literal["cpu", "cuda"],
            **kwargs
        ) -> None:
        self.tokenizer = RobertaForTokenClassification.from_pretrained(tokenizer_path, **kwargs).to(device)
        return None
    
    def load_model(
            self, 
            model_path: Union[str, Path],
            device: Literal["cpu", "cuda"],
            num_labels: Optional[int] = 9,
            **kwargs
        ) -> None:
        self.model = RobertaForTokenClassification.from_pretrained(model_path, num_labels = num_labels, **kwargs).to(device)
        return None

    def prepare_dataset(self) -> None:
        assert(self.tokenizer != None), "Load the tokenizer and model first"

        def tokenize_and_align_labels(examples, label_all_tokens = True):
            tokenized_inputs = self.tokenizer(
                examples["tokens"], 
                truncation = True, 
                is_split_into_words = True,
                return_tensors = "pt"
            )

            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index = i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    previous_word_idx = word_idx

                label.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        self.tokenized_dataset = self.dataset.map(tokenize_and_align_labels, batched = True)

        self.labels = self.dataset["train"].features["ner_tags"].feature.names

        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        
        return None
    
    def compute_metrics(self, p) -> Dict[str, float]:
        if not self.metric:
            self.metric = load_metric("seqeval")

        predictions, labels = p
        predictions = np.argmax(predictions, axis = 2)

        true_preds = [
            [self.labels[p] for (p,l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [self.labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions = true_preds, references = true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }
    
    def train(
            self,
            output_dir: Union[str, Path],
            optimizers: Dict[str, Tuple[optim.Optimizer, Dict[str, Any]]], 
            schedulers: Dict[str, Tuple[optim.lr_scheduler.LambdaLR,]],
            tokenizer_path: Union[str, Path],
            model_path: Union[str, Path],
            epochs: int,
            batch_size: int,
            result_csv: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> pd.DataFrame:
        assert(self.model), "Instantiate a model"
        assert(self.tokenizer), "Instantiate a tokenizer"
        assert(self.tokenized_dataset), "Tokenize training dataset"
        
        self.metric = load_metric("seqeval")

        try: 
            df = pd.read_csv(result_csv)
        except:
            df = pd.DataFrame(
                np.zeros(len(optimizers.keys(),len(schedulers.keys()))), 
                index = list(optimizers.keys()),
                columns = list(schedulers.keys())
            )
            df.to_csv(result_csv)

        for optimizer, optimizer_params in optimizers.items():
            for scheduler, scheduler_params in schedulers.items():
                
                self.load_model(model_path, device)
                self.load_tokenizer(tokenizer_path, len(self.labels), device)

                self.data_collator = DataCollatorForTokenClassification(self.tokenzier)
                
                args = TrainingArguments(
                    output_dir = f"{output_dir}/{optimizer}_{scheduler}",
                    evaluation_strategy = "epoch",
                    per_device_train_batch_size = batch_size,
                    per_device_eval_batch_size = batch_size,
                    num_train_epochs = epochs,
                    push_to_hub = False
                )

                t0 = time()

                trainer = Trainer(
                    self.model,
                    args,
                    train_dataset = self.tokenized_dataset["train"],
                    eval_dataset = self.tokenized_dataset["validation"],
                    tokenizer = self.tokenizer,
                    data_collator = self.data_collator,
                    compute_metrics = self.compute_metrics,
                    optimizer = (optimizer, scheduler)
                )

                trainer.train()

                predictions, labels, _ = trainer.predict(self.tokenized_dataset["test"])
                labels = labels.detach().to("cpu")
                predictions = np.argmax(predictions.detach().to("cpu"), axis = 2)

                true_predictions = [
                    [self.labels[p] for (p,l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]

                true_labels = [
                    [self.labels[l] for (p,l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels) 
                ]

                results = self.metric.compute(predictions = true_predictions, references = true_labels)
                results["train_time"] = time() - t0

                df[optimizer,scheduler] = results
                df.to_csv(result_csv)

        return df
    
    def infer(self):
        assert (self.model and self.tokenizer), "Load a trained tokenizer and model"

        return None
    

class SkinCancer(BaseDataset):

    def __init__(self):
        super().__init__()

        self.dataset_name: str = "SkinCancer"

        self.dataset = load_dataset(
            "marmal88/skin_cancer"
        )
        self.processed_dataset = None

        self.model = None
        self.feature_extractor = None
        
        self.metric = load_metric("accuracy")
        
        self.label2id = dict()
        self.id2label = dict()

    def prepare_dataset(self) -> None:
        
        labels = list(set(self.dataset["train"]["dx"]))
        
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        
        def preprocess(batch, device: Literal["cpu", "cuda"]):
            inputs = self.feature_extractor(
                batch["image"],
                return_tensors = "pt"
            )
            
            inputs["label"] = batch["dx"]
            return inputs
        
        self.processed_dataset = self.dataset.with_transform(preprocess)
        
        return None
    
    def data_collator(batch):
        return {
                "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
                "labels": torch.tensor([x["label"] for x in batch])
            }
    
    def load_model(
            self, 
            model_path: Union[str, Path],
            device: Literal["cpu", "cuda"],
            labels: int
        ) -> None:
        self.model = MobileViTV2ForImageClassification.from_pretrained(model_path, num_labels = labels).to(device)
        
        return None
    
    def load_feature_extractor(
            self, 
            feature_extractor_path: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> None:

        self.feature_extractor = MobileViTFeatureExtractor.from_pretrained(feature_extractor_path).to(device)

    
    def compute_metrics(self, p):
        return self.metric.compute(
            predictions = np.argmax(p.predictions, axis = 1),
            references = p.label_ids
        )
    
    def train(
            self,
            output_dir: Union[str, Path],
            optimizers: Dict[Any, Any],
            schedulers: Dict[Any, Any],
            feature_extractor_path: Union[str, Path],
            model_path: Union[str, Path],
            epochs: int,
            batch_size: int,
            result_csv: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> pd.DataFrame:

        assert(self.model), "Instantiate a model"
        assert(self.tokenizer), "Instantiate a tokenizer"
        assert(self.tokenized_dataset), "Tokenize the training dataset"

        try: 
            df = pd.read_csv(result_csv)
        except:
            df = pd.DataFrame(
                np.zeros(len(optimizers.keys(),len(schedulers.keys()))), 
                index = list(optimizers.keys()),
                columns = list(schedulers.keys())
            )
            df.to_csv(result_csv)
        
        for optimizer, optimizer_params in optimizers.items():
            for scheduler, scheduler_params in schedulers.items():

                self.load_model(model_path, device, len(self.id2label))
                self.load_feature_extractor(feature_extractor_path, device)

                args = TrainingArguments(
                    output_dir = f"{output_dir}/{optimizer}_{scheduler}",
                    evaluation_strategy = "epoch",
                    per_device_train_batch_size = batch_size,
                    per_device_eval_batch_size = batch_size,
                    num_train_epochs = epochs,
                    push_to_hub = False,
                    
                )

                t0 = time()

                trainer = Trainer(
                    self.model,
                    args,
                    train_dataset = self.processed_dataset["train"],
                    eval_dataset = self.processed_dataset["validation"],
                    tokenizer = self.feature_extractor,
                    data_collator = self.data_collator,
                    compute_metrics = self.compute_metrics,
                    optimizer = (optimizer, scheduler)
                )

                trainer.train()
                
                predictions, labels, _ = trainer.predict(self.tokenized_dataset["text"])
                labels = labels.detach().to("cpu")
                predictions = predictions.detach().to("cpu")
                
                results = self.metric.compute(
                    predictions = np.argmax(predictions, axis = 1),
                    references = labels
                )
                
                results["train_time"] = time() - t0
                df[optimizer, scheduler] = results
                df.to_csv(result_csv)
                       
        return df