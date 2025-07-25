from transformers.models.deberta_v2 import DebertaV2ForTokenClassification
import copy

class ModelInit:
    def __init__(
        self,
        checkpoint: str,
        id2label: dict,
        label2id: dict,
        freeze_embedding: bool,
        freeze_layers: int,
    ) -> None:
        self.model = DebertaV2ForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        for param in self.model.deberta.embeddings.parameters():
            param.requires_grad = False if freeze_embedding else True
        for layer in self.model.deberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        self.weight = copy.deepcopy(self.model.state_dict())

    def __call__(self) -> DebertaV2ForTokenClassification:
        self.model.load_state_dict(self.weight)
        return self.model

