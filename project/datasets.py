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

import transformers
from transformers import (
    RobertaForTokenClassification, 
    RobertaTokenizerFast,
    MobileViTForImageClassification,
    MobileViTImageProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    DefaultDataCollator
)

from torchvision.transforms import v2

import datasets
from datasets import load_dataset
import evaluate

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
            **kwargs
        ) -> None:
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, add_prefix_space = True, **kwargs)
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
                is_split_into_words = True
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

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        self.tokenized_dataset = self.dataset.map(tokenize_and_align_labels, batched = True)

        self.labels = self.dataset["train"].features["ner_tags"].feature.names

        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        
        return None
    
    def compute_metrics(self, p) -> Dict[str, float]:
        if not self.metric:
            self.metric = evaluate.load("seqeval")

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
            optimizers: Dict[str, Dict[str,Any]], 
            schedulers: Dict[str, Dict[str,Any]],
            tokenizer_path: Union[str, Path],
            model_path: Union[str, Path],
            strategy: Literal["epoch", "steps"],
            epochs: int,
            batch_size: int,
            result_csv: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> pd.DataFrame:
        
        assert(self.labels), "Labels not found"
        
        self.load_tokenizer(tokenizer_path)
        self.metric = evaluate.load("seqeval")
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        df = pd.DataFrame(
            index = list(optimizers.keys()),
            columns = list(schedulers.keys())
        )
        df.to_csv(result_csv)

        for optimizer_name, (optimizer, optimizer_params) in optimizers.items():
            for scheduler_name, scheduler_params in schedulers.items():
                
                print("----------------------------")
                print(f"Optimizer: {optimizer_name} | Scheduler: {scheduler_name}")
                _scheduler_params = copy(scheduler_params)
                
                self.load_model(model_path, device)

                self.model.config.id2label = self.id2label
                self.model.config.label2id = self.label2id
                
                if optimizer == torch.optim.RMSprop:
                    additional_args = {"weight_decay": 0.}
                else:
                    additional_args = dict()
                
                args = TrainingArguments(
                    output_dir = f"{output_dir}/{optimizer_name}_{scheduler_name}",
                    evaluation_strategy = strategy,
                    per_device_train_batch_size = batch_size,
                    per_device_eval_batch_size = batch_size,
                    num_train_epochs = epochs,
                    push_to_hub = False,
                    disable_tqdm = True,
                    **additional_args
                )
                
                _optimizer = optimizer(self.model.parameters(), **optimizer_params)
                _scheduler = transformers.get_scheduler(
                    scheduler_name, 
                    _optimizer, 
                    num_warmup_steps = _scheduler_params.pop("num_warmup_steps"),
                    num_training_steps = epochs*len(self.tokenized_dataset["train"])//batch_size,
                    scheduler_specific_kwargs = _scheduler_params
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
                    optimizers = (_optimizer, _scheduler)
                )

                trainer.train()

                predictions, labels, _ = trainer.predict(self.tokenized_dataset["test"])
                labels = labels
                predictions = np.argmax(predictions, axis = 2)

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
                results["log"] = trainer.state.log_history

                df.at[optimizer_name,scheduler_name] = results
                df.to_csv(result_csv)
                
                del _optimizer
                del _scheduler
                del self.model
                del trainer

        torch.cuda.empty_cache()
        
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
        self.image_processor = None
        
        self.metric = evaluate.load("accuracy")
        
        self.labels = list()
        self.label2id = dict()
        self.id2label = dict()

    def prepare_dataset(self) -> None:
        
        self.labels = list(set(self.dataset["train"]["dx"]))
        
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        size = (
            self.image_processor.size["shortest_edge"] 
            if "shortest_edge" in self.image_processor.size
            else (self.image_processor.size["height"], self.image_processor.size["width"])
        )
        
        self._transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale = True),
            v2.RandomResizedCrop(size), 
            v2.ToDtype(torch.float32, scale = True),
            v2.GaussianBlur(kernel_size = (5,9), sigma = (0.1,5)),
            v2.ColorJitter(brightness = .5, contrast = .5, saturation = .5),
            v2.RandomRotation(degrees = (0,180))
        ])
        
        def transforms(examples):
            examples["pixel_values"] = [self._transforms(img.convert("RGB")) for img in examples["image"]]
            del examples["image"]
            del examples["age"]
            del examples["image_id"]
            del examples['lesion_id']
            del examples["dx_type"]
            del examples["sex"]
            del examples["localization"]
            examples["labels"] = [self.label2id[label] for label in examples["dx"]]
            del examples["dx"]
            return examples
        
        self.processed_dataset = self.dataset.with_transform(transforms)
        
        return None
    
    def load_model(
            self, 
            model_path: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> None:
        assert(self.labels), "Load dataset"
        
        self.model = MobileViTForImageClassification.from_pretrained(
            model_path, 
            num_labels = len(self.labels),
            ignore_mismatched_sizes = True
        ).to(device)
        
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        
        return None
    
    def load_image_processor(
            self, 
            image_processor_path: Union[str, Path]
        ) -> None:

        self.image_processor = MobileViTImageProcessor.from_pretrained(image_processor_path)
        
        return None

    
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis = 1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def train(
            self,
            output_dir: Union[str, Path],
            optimizers: Dict[Any, Any],
            schedulers: Dict[Any, Any],
            image_processor_path: Union[str, Path],
            model_path: Union[str, Path],
            strategy: Literal["epoch", "steps"],
            epochs: int,
            batch_size: int,
            result_csv: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> pd.DataFrame:

        assert(self.labels), "Labels not found"
        
        self.data_collator = DefaultDataCollator()
        
        self.load_image_processor(image_processor_path)

        df = pd.DataFrame(
            index = list(optimizers.keys()),
            columns = list(schedulers.keys())
        )
        df.to_csv(result_csv)
        
        for optimizer_name, (optimizer, optimizer_params) in optimizers.items():
            for scheduler_name, scheduler_params in schedulers.items():


                print("----------------------------")
                print(f"Optimizer: {optimizer_name} | Scheduler: {scheduler_name}")
                _scheduler_params = copy(scheduler_params)
                
                self.load_model(model_path, device)

                if optimizer == torch.optim.RMSprop:
                    additional_args = {"weight_decay": 0.}
                else:
                    additional_args = dict()
                
                args = TrainingArguments(
                    output_dir = f"{output_dir}/{optimizer_name}_{scheduler_name}",
                    evaluation_strategy = strategy,
                    per_device_train_batch_size = batch_size,
                    per_device_eval_batch_size = batch_size,
                    num_train_epochs = epochs,
                    push_to_hub = False,
                    remove_unused_columns = False,
                    #disable_tqdm = True,
                    **additional_args
                )

                _optimizer = optimizer(self.model.parameters(), **optimizer_params)
                _scheduler = transformers.get_scheduler(
                    scheduler_name, 
                    _optimizer, 
                    num_warmup_steps = _scheduler_params.pop("num_warmup_steps"),
                    num_training_steps = epochs*len(self.processed_dataset["train"])//batch_size,
                    scheduler_specific_kwargs = _scheduler_params
                )

                t0 = time()

                trainer = Trainer(
                    self.model,
                    args,
                    train_dataset = self.processed_dataset["train"],
                    eval_dataset = self.processed_dataset["validation"],
                    tokenizer = self.image_processor,
                    data_collator = self.data_collator,
                    compute_metrics = self.compute_metrics,
                    optimizers = (_optimizer, _scheduler)
                )

                trainer.train()
                
                predictions, labels, _ = trainer.predict(self.processed_dataset["test"])
                
                results = self.metric.compute(
                    predictions = np.argmax(predictions, axis = 1),
                    references = labels
                )
                
                results["train_time"] = time() - t0
                results["log"] = trainer.state.log_history
                df[optimizer_name, scheduler_name] = results
                df.to_csv(result_csv)
                
                del _optimizer
                del _scheduler
                del self.model
                del trainer
                       
        torch.cuda.empty_cache()               
        
        return df