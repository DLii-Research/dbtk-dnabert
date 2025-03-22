from dbtk.data.datasets import SequenceDataset, SequenceTaxonomyDataset
from dbtk.data.vocabularies import Vocabulary
from dbtk.data import transforms
from dbtk.data.transforms.compositions import DnaSequenceTransform
from dnadb import fasta, taxonomy
import lightning as L
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Union

from .tokenizers import DnaTokenizer

class DnaBertPretrainingDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer: DnaTokenizer,
        train_sequences_path: Union[str, Path],
        test_sequences_path: Optional[Union[str, Path]] = None,
        val_split: float = 0.0,
        min_length: int = 65,
        max_length: int = 250,
        reverse_complement: bool = True,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_sequences_path = train_sequences_path
        self.test_sequences_path = test_sequences_path
        self.val_split = val_split
        self.min_length = min_length
        self.max_length = max_length
        self.reverse_complement = reverse_complement
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            lambda fasta_entry: fasta_entry.sequence,
            DnaSequenceTransform(
                min_length=self.min_length,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                reverse_complement=self.reverse_complement,
                pad_token_id=self.tokenizer.vocab["[PAD]"]
            )
        ])

    def collate(self, batch):
        batch = torch.stack(batch)
        return {
            "kmers": batch
        }

    def setup(self, stage: str):
        if stage == "fit":
            train_sequences = SequenceDataset(
                self.train_sequences_path,
                transform=self.transform
            )
            self.train_sequences, self.val_sequences = torch.utils.data.random_split(
                train_sequences,
                [1.0 - self.val_split, self.val_split],
                generator=torch.Generator()
            )

        elif stage == "test":
            self.test_sequences = SequenceDataset(
                self.test_sequences_path,
                transform=self._transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_sequences,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_sequences,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_sequences,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            shuffle=False,
            num_workers=self.num_workers
        )


class DnaBertTaxonomyDataModule(L.LightningDataModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer: Callable,
        train_sequences_path: Union[str, Path],
        train_taxonomies_path: Union[str, Path],
        test_sequences_path: Optional[Union[str, Path]] = None,
        test_taxonomies_path: Optional[Union[str, Path]] = None,
        val_split: float = 0.0,
        min_length: int = 65,
        max_length: int = 250,
        kmer: int = 1,
        kmer_stride: int = 1,
        mask_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.train_sequences_path = train_sequences_path
        self.train_taxonomies_path = train_taxonomies_path
        self.test_sequences_path = test_sequences_path
        self.test_taxonomies_path = test_taxonomies_path
        self.val_split = val_split
        self.min_length = min_length
        self.max_length = max_length
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _sequence_transform(self, fasta_entry: fasta.FastaEntry):
        # Sequence
        sequence = fasta_entry.sequence

        # Randomly trim the sequence such that the resulting kmer-sequence length
        # is within the minimum/maximum sequence length range.
        min_length = min(self.min_length*self.kmer_stride - 1 + self.kmer, len(sequence))
        max_length= min(self.max_length*self.kmer_stride - 1 + self.kmer, len(sequence))
        length = torch.randint(min_length, max_length, size=(1,)).item()
        offset = torch.randint(0, len(sequence) - length + 1, size=(1,)).item()
        sequence = torch.tensor(list(self.vocabulary(self.tokenizer(sequence[offset:offset+length]))))

        # Padding
        sequence = F.pad(sequence, (0, self.max_length - len(sequence)), value=self.vocabulary["[PAD]"])

        return sequence

    def _taxonomy_transform(self, taxonomy_entry: taxonomy.TaxonomyDbEntry):
        return torch.tensor(taxonomy_entry.taxonomy.taxonomy_ids)

    def _collate(self, batch):
        sequences, taxonomies = zip(*batch)
        sequences = torch.stack(sequences)
        taxonomies = torch.stack(taxonomies)
        taxonomies = taxonomies.permute(-1, *torch.arange(taxonomies.ndim - 1))
        return sequences, taxonomies

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = SequenceTaxonomyDataset(
                self.train_sequences_path,
                self.train_taxonomies_path,
                sequence_transform=self._sequence_transform,
                taxonomy_transform=self._taxonomy_transform
            )
            num_val_data = int(len(self.train_data) * self.val_split)
            self.train_data, self.val_data = torch.utils.data.random_split(
                self.train_data,
                [len(self.train_data) - num_val_data, num_val_data],
                generator=torch.Generator()
            )

        elif stage == "test":
            self.test_data = SequenceTaxonomyDataset(
                self.test_sequences_path,
                self.test_taxonomies_path,
                sequence_transform=self._sequence_transform,
                taxonomy_transform=self._taxonomy_transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self.num_workers
        )