import requests
import tensorflow as tf
import os

# URL for Alice's Adventures in Wonderland
url = 'https://www.gutenberg.org/files/11/11-0.txt'
# Fetch the text
response = requests.get(url)
text = response.text.lower()  # Convert text to lowercase

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])

# Preparing the dataset
seq = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1
seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(seq)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(10000).batch(64, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256),
    tf.keras.layers.LSTM(1024, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Define the checkpoint directory and filename
checkpoint_dir = 'C:/Users/Scyp/Code/Gen AI/checkpoints/'

# Attempt to load the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    model.load_weights(latest_checkpoint)
    print(f"Loaded weights from {latest_checkpoint}")
else:
    print("No checkpoint found, starting training from scratch.")

# Define the ModelCheckpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5"),
    save_weights_only=True,
    monitor='loss',
    verbose=1,
    save_best_only=True)

def cleanup_checkpoints(checkpoint_dir, max_to_keep=5):
    """Keep only the most recent `max_to_keep` checkpoints in the directory."""
    checkpoints = sorted(
        [os.path.join(checkpoint_dir, fname) for fname in os.listdir(checkpoint_dir) if fname.endswith('.h5')],
        key=os.path.getmtime,
        reverse=True
    )
    for old_checkpoint in checkpoints[max_to_keep:]:
        os.remove(old_checkpoint)
        print(f"Removed old checkpoint: {old_checkpoint}")

# After defining and compiling your model, and setting up your dataset

# Train the model
EPOCHS = 15  # Adjust the number of epochs based on your requirements
# Now, pass this callback to the model's fit method
model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# After training, you can clean up old checkpoints
cleanup_checkpoints(checkpoint_dir, max_to_keep=5)

# Convert text to a sequence of integers
seq = tokenizer.texts_to_sequences([text])[0]

# Vocabulary size (needed for the model's last layer)
vocab_size = len(tokenizer.word_index) + 1

# Define the sequence length
seq_length = 100  # Length of the sequence to be fed into the model
examples_per_epoch = len(text) // (seq_length + 1)

# Create training sequences
char_dataset = tf.data.Dataset.from_tensor_slices(seq)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]  # Input sequence
    target_text = chunk[1:]  # Target sequence, shifted by one
    return input_text, target_text

# Use the map method to apply the split function to all sequences
dataset = sequences.map(split_input_target).batch(64, drop_remainder=True)
dataset = dataset.shuffle(10000).batch(64, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# Batch size
BATCH_SIZE = 64

# Buffer size for shuffling
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Prefetching for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

def generate_text(model, start_string, generation_length=1000):
    # Convert start_string to numbers (vectorizing)
    input_eval = [tokenizer.word_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    for i in range(generation_length):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Use a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(tokenizer.index_word.get(predicted_id, ''))

    return (start_string + ''.join(text_generated))

# Note: Ensure you train your model here before generating text
# model.fit(dataset, epochs=EPOCHS)

# Generating text (make sure the model is trained or this is for testing purposes)
generated_text = generate_text(model, start_string="alice ", generation_length=500)
print(generated_text)