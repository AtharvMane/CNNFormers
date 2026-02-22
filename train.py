import datasets
import transformers
import evaluate
from transformers import TrainingArguments, Trainer

from model.transcnn import TransResNet


# Metrics
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Called by the Trainer to compute metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)



# function to apply transforms
def apply_transforms(examples, transform):
    examples['pixel_values'] = [train_transforms(transforms.ToTensor()(image.convert('RGB'))) for image in examples['image']]
    examples['labels'] = examples.pop('label')
    del examples['features']
    del examples["image"]
    return examples



if __name__=="__main__":
	# Load the "320px" subset of Imagenette
	ds = datasets.load_dataset("data/essence_of_imagenet")

	# Get label info
	labels_list = ds['train'].features['label'].names
	num_labels = len(labels_list)
	print(f"Found {num_labels} labels: {labels_list}")

	train_transforms = transforms.Compose([
    		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	val_transforms = transforms.Compose([
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	# Apply preprocessing
	train_ds = ds['train'].with_transform(lambda x: apply_transforms(x, train_transforms))
	val_ds = ds['validation'].with_transform(lambda x: apply_transforms(x, val_transforms))

	training_args = TrainingArguments(
		output_dir="./checkpoints",
		per_device_train_batch_size=8,
		per_device_eval_batch_size=8,
		eval_strategy="epoch",     # Run evaluation every epoch
		save_strategy="epoch",            # Save checkpoint every epoch
		report_to="wandb",
		num_train_epochs=12,               # Total number of epochs (use more for real training)
		learning_rate=1e-3,
		load_best_model_at_end=True,      # Load the best model at the end
		metric_for_best_model="accuracy", # Use accuracy to find the best model
		logging_dir='./logs',
		logging_steps=50,
		warmup_steps=100,
		run_name="cnnformer_bigger_batch_Run_essence_of_imageneti_local",
		# This is CRUCIAL. By default, Trainer removes columns
		# not used by the model's forward() signature.
		# We need to keep the 'label' column for our wrapper's logic.
		remove_unused_columns=False,
	)
