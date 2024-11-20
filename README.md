# AI-powered Newsreader

## Introduction
This repository is an end-to-end implementation of a simple AI-powered newsreader. The goal is to:
- Provide content from curated list of sources
- The content should be filtered / ordered based on an algorithmic rating of "quality"
- The algorithm should learn over time as preferences and content changes
- The content should be presented with context (summary, topics, etc) to  help the user determine if something is worthwhile to consume

Refer to ["Building a Personalized News Reader with AI"](https://benjamintseng.com/portfolio/building-a-personalized-news-reader-with-ai/) for more information on the project architecture, technical choices, and motivation.

## Dependencies
The repository includes the core files necessary but requires a few additional elements to setup:
1. A Postgres database with [`pgvector`](https://github.com/pgvector/pgvector) extension setup — refer to ["Building a Personalized News Reader with AI"](https://benjamintseng.com/portfolio/building-a-personalized-news-reader-with-ai/) for the tables and indices needed. I've utilized [Supabase](https://supabase.com/) but, in principle, any Postgres service that supports `pgvector` will work
2. [Modal.com](https://modal.com/) — This uses Modal's serverless platform to train and serve the model, carry out the scrape processes necessary, and serve the web application. The individual Python files become  Modal functions which are invoked on a regular basis via [Modal's cron functionality](https://modal.com/docs/guide/cron) or by other Modal functions. The easy integration with the command line and Python make this a very fast way of deploying novel services. In particular, it makes use of:
    * [Modal secrets](https://modal.com/docs/guide/secrets) which pass secret parameters (like API keys, connection strings, etc) as environment variables
    * [Modal volumes](https://modal.com/docs/guide/volumes) which allow persistent storage of templates, parameters, and model weights. For this project, I have a simple directory structure with web application-specific templates and parameters in `/api/` directory, model-specific weights and parameters in `/model/` directory, and the crawl record and scraping parameters in `/scrape/` directory
    * [Modal queue](https://modal.com/docs/guide/dicts-and-queues) which allow for short-term storage of data to communicate between different remote jobs
3. An initial model — the first versions of the Keras model were trained on Google CoLab using a similar `tf.data` pipeline to what's in the `train()` function in `modal_train.py`. The below Keras model corresponding to the architecture listed out on ["Building a Personalized News Reader with AI"](https://benjamintseng.com/portfolio/building-a-personalized-news-reader-with-ai/) was employed. This was then loaded onto a Modal volume for inference and further fine-tuning.
4. Scrapers — The sample code here only provides example scrapers for Google and Huggingface blogs as examples of how the architecture works. For other sites (or if Google/Huggingface dramatically revise their web page designs), you will need to supply your own scrapers. 

## Keras Model
`modal_train.py` and `modal_recommend.py` assume the availabilty of a pre-trained model that conforms to the following Keras code. The `tf.data` data pipeline and the model compilation / fitting / evaluation used in  the `modal_train.py` are identical to what was used to initially train this model on Google CoLab.

```
# needs jax 0.4 and Keras 3
import keras_hub
import keras
import numpy as np

# parameters
intermediate_dim = 128
roberta_dim = 768
text_embedding_dim = roberta_dim*2
user_embedding_dim = 500
user_intermediate_dim = 512
batch_size = 10
num_users = 10
max_chunk_size = 512
train_val_split = 0.1

# pull Roberta preprocessor
model_id = "roberta_base_en"
preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(model_id)
tokenizer = preprocessor.tokenizer

# pull Roberta backbone
backbone = keras_hub.models.Backbone.from_preset(model_id)

# assumes you've set up a tf.data pipeline similar to what's in modal_train.py
# define backbone model
input_tokens = keras.Input(shape=(max_chunk_size,), dtype='int32')
input_padding = keras.Input(shape=(max_chunk_size,), dtype="bool")
x = backbone({'token_ids':input_tokens, 'padding_mask':input_padding})
x = keras.layers.GlobalAveragePooling1D()(x, input_padding)
x = keras.layers.Dropout(0.5)(x)
emb = keras.layers.Dense(text_embedding_dim, activation='tanh')(x)
backbone_model = keras.Model(inputs=[input_tokens, input_padding], outputs=emb, name='backbone_model')

# define length model
lenx = keras.layers.Dropout(0.5)(emb)
lenx = keras.layers.Dense(intermediate_dim, activation='relu', kernel_regularizer='l2')(lenx)
lenout = keras.layers.Dense(1, name='length_out')(lenx)

# define recommendation model
input_user = keras.Input(shape=(1,), dtype='int32', name='user_id')
user_emb = keras.layers.Flatten()(
    keras.layers.Embedding(num_users, user_embedding_dim, name='user_preferences')(input_user)
)
recx = keras.layers.Dot(axes=-1)([emb, user_emb])
recout = keras.ops.sigmoid(recx)

# define model for training
train_model = keras.Model(
    inputs=[input_tokens, input_padding, input_user], 
    outputs=[lenout, recout], 
    name='train_model'
)
```