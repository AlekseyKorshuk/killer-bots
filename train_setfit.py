from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer

# Load a dataset from the Hugging Face Hub
dataset = load_dataset("AlekseyKorshuk/is_title")

print(dataset)

# Simulate the few-shot regime by sampling 8 examples per class
num_classes = 2
train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["validation"]

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=1,  # The number of epochs to use for constrastive learning
    # column_mapping={"sentence": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# Push model to the Hub
trainer.push_to_hub("AlekseyKorshuk/is-title-setfit")
