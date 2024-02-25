import os
import json
from time import time

from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn, optim

from transformers import (
    RobertaForTokenClassification, 
    RobertaTokenizerFast,
    RobertaConfig,
    MobileViTV2ForImageClassification,
    MobileViTImageProcessor,
    MobileViTV2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

import datasets
from datasets import load_dataset, load_metric

from utils import create_dir, set_seed
from model import RoadSafetyNet

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
    



class RoadSafety(BaseDataset):

    def __init__(
            self,
            model: Optional[nn.Module] = None
        ) -> None:
        super().__init__()

        self.dataset_name: str = "road-safety"

        self.dataset = load_dataset(
            path = "inria-soda/tabular-benchmark",
            data_files = "clf_cat/road-safety.csv"
        )

        model: nn.Module = model


    def prepare_dataset(
            self, 
            split: Optional[float] = 0.2, 
            seed: Optional[int] = 1234
        ) -> None:

        dataset = self.dataset.train_test_split(split = split, seed = seed)
        self.Train = dataset["train"]
        self.Test = dataset["test"]

        del dataset

        return None
    
    def load_model(self, device:Literal["cpu", "cuda"], model: Optional[nn.Module] = RoadSafetyNet) -> None:
        self.model = model().to_device(device)
        return None

    def train_one_epoch(self, optimizer: optim.Optimizer, batch_size: int) -> float:
        avg_loss = 0

        idxs = np.arange(0, len(self.Train), batch_size)

        for i, idx in enumerate(idxs):
            
            optimizer.zero_grad()
            data = self.Train[idx:idx+batch_size]

            outputs = self.model(list(data.values())[:-1]).detach().to_cpu()
            loss = self.loss(outputs, data["SexofDriver"])

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if i%batch_size == batch_size-1:
                last_loss = avg_loss/(len(data["SexofDriver"]))
                print(f"\tbatch {i+1} | loss: {last_loss}")
                avg_loss = 0.

        return last_loss
    
    def test_one_epoch(self, batch_size: int) -> float:
        self.model.eval()
        avg_loss = 0.

        idxs = np.arange(0,len(self.Test),batch_size)

        with torch.no_grad():
            for i, idx in enumerate(idxs):               
                data = self.Test[idx:idx+batch_size]

                outputs = self.model(list(data.values())[:-1]).detach().to_cpu()
                loss = self.loss(outputs, data["SexofDriver"])

                avg_loss += loss.item()

                if i%batch_size == batch_size-1:
                    last_loss = avg_loss/len(data["SexofDriver"])
                    print(f"\tbatch {i+1} | loss: {last_loss}")
                    avg_loss = 0.
                
        return last_loss
                

    def train(
            self,
            result_csv: Union[str, Path],
            optimizers: Dict[str, Tuple[optim.Optimizer, Dict[str, Any]]],
            schedulers: Dict[str, Tuple[optim.lr_scheduler._LRScheduler, Dict[str, Any]]],
            loss_func: Optional[nn.modules.loss._Loss] = nn.BCELoss,
            epochs: Optional[int] = 50,
            batch_size: Optional[int] = 20,
            model: Optional[nn.Module] = RoadSafetyNet,
            device: Optional[Literal["cpu", "cuda"]] = "cpu"
        ) -> pd.DataFrame:

        self.loss: nn.modules.loss._Loss = loss_func

        try:
            df = pd.read_csv(result_csv)
        except:
            df = pd.DataFrame(
                np.zeros((len(optimizers),len(schedulers))), 
                index = list(optimizers.keys()), 
                columns = list(schedulers.keys())
            )
            df.to_csv(result_csv)

        for scheduler_name, (scheduler, scheduler_params) in schedulers.items():
            for optimizer_name, (optimizer, optimizer_params) in optimizers.items():
                t0 = time()

                self.load_model(model, device)

                print(f"|{scheduler_name}, {optimizer_name}|")
                optimizer = optimizer(**optimizer_params)
                scheduler = scheduler(optimizer, **scheduler_params)
                
                best_loss = np.nan

                run_data = {"train_loss": [], "test_loss": [], "train_time": 0.,}

                for epoch in range(epochs):
                    print(f"Epoch {epoch+1}")
                    self.model.train(True)

                    avg_train_loss = self.train_one_epoch(optimizer, batch_size)

                    avg_test_loss = self.test_one_epoch(batch_size)

                    run_data["train_loss"].append(avg_train_loss)
                    run_data["test_loss"].append(avg_test_loss)

                    if avg_test_loss < best_loss:
                        best_loss = avg_test_loss
                        model_path = f"/models/road_safety/{optimizer_name}_{scheduler_name}"
                        torch.save(self.model.state_dict(), model_path)
                
                    scheduler.step()

                run_data["train_time"] = time() - t0
                df[optimizer_name,scheduler_name] = run_data
                df.to_csv(result_csv)

                self.model.to_cpu()
            
        return df
    
    def get_model(self) -> Union[None, nn.Module]:
        if self.model:
            return self.model
        
        else:
            print("No model trained")
            return None

    def infer(self, data, model: nn.Module, batch_size: int, device: Literal["cpu", "cuda"]) -> tuple[np.ndarray, float]:
        t0 = time()
        N = len(data["SexofDriver"])

        preds = np.zeros((N))
        idxs = np.arange(0,N, batch_size)

        self.model.to_device(device)

        print("Beginning inference")
        with torch.no_grad():
            for i, idx in enumerate(idxs):
                print(f"\tbatch {i+1}")
                preds[idx:np.min(idx+batch_size,N)] = model(list(data[idx:idx+batch_size].values())[:-1]).detach().to_cpu()

        return preds, np.mean(int(preds) == data["SexofDriver"].values()), (time()-t0)/N
    

    

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

    def load_tokenizer(
            self,
            tokenizer_path: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> None:
        self.tokenizer = RobertaForTokenClassification.from_pretrained(tokenizer_path).to_device(device)
        return None
    
    def load_model(
            self, 
            model_path: Union[str, Path],
            device: Literal["cpu", "cuda"],
            num_labels: int
        ) -> None:
        self.model = RobertaForTokenClassification.from_pretrained(model_path, num_labels = num_labels).to_device(device)
        return None

    def prepare_dataset(self) -> None:
        assert(self.tokenizer != None), "Load the tokenizer and model first"

        def tokenize_and_align_labels(examples, label_all_tokens = True):
            tokenized_inputs = self.tokenizer(examples["tokens"], truncation = True, is_split_into_words = True)

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
            optimizers: Dict[Any, Any], 
            schedulers: Dict[Any, Any],
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
                    output_dir = "",
                    evaluation_strategy = "epoch",
                    per_device_train_batch_size = batch_size,
                    per_device_eval_batch_size = batch_size,
                    num_train_epochs = epochs
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
            "marmal88/skin-cancer"
        )

        self.model = None
        self.image_processor = None

    def prepare_dataset(self) -> None:
        return None
    
    def load_model(
            self, 
            model_path: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> None:
        self.model = MobileViTV2ForImageClassification.from_pretrained(model_path).to_device(device)
        
        return None
    
    def load_image_process(
            self, 
            image_processor_path: Union[str, Path],
            device: Literal["cpu", "cuda"]
        ) -> None:

        self.image_processor = MobileViTImageProcessor.from_pretrained(image_processor_path).to_device(device)

        return None
    
    def train(
            self,
            optimizers: Dict[Any, Any],
            schedulers: Dict[Any, Any],
            image_processor_path: Union[str, Path],
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

                self.load_model(model_path, device)
                self.load_image_process(image_processor_path, device)


        
        return df