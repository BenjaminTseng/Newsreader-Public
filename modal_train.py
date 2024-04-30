import modal

tf_image = (
    modal.Image.from_registry('tensorflow/tensorflow:2.16.1-gpu')
    .pip_install('keras==3.3.2')
    .pip_install('keras-nlp==0.9.3')
    .pip_install('psycopg2-binary')
    .pip_install('pgvector')
)

app = modal.App("newsreader-train", image=tf_image)
vol = modal.Volume.from_name("newsreader-data")

type_ALL = 1 # train all layers

# process run twice a month
# type currently unnecessary variable but can be used in the future to change training types
@app.function(gpu="L4", timeout=10800, volumes={"/data": vol}, secrets=[modal.Secret.from_name("newsreader_psycopg2")], schedule=modal.Cron("30 9 14,28 * *"))
def train(type=type_ALL):
    # eliminate always there tensorflow warnings from the logs
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import warnings 
    warnings.filterwarnings('ignore')

    import keras
    import keras_nlp
    import tensorflow as tf
    import json
    import numpy as np
    import datetime
    import psycopg2
    import psycopg2.extras

    print('beginning training run', datetime.datetime.now())
    vol.reload()

    # file for training parameters
    with open('/data/model/train_params.json', 'r') as f:
        trainparams = json.load(f)
    
    # pull parameters
    batch_size = trainparams['batch_size']
    train_val_split = trainparams['train_val_split']
    model_id = trainparams['model_id']
    max_user_ratings = trainparams['max_user_ratings']
    type_all_epochs = trainparams['type_all_epochs']
    recommend_mse_patience = trainparams['recommend_mse_patience']

    # path for model
    with open('/data/model/best_model.txt', 'r') as f:
        model_path = '/data/model/'+f.read().strip()

    # pull model
    print('loading model', model_path)
    model = keras.models.load_model(model_path, compile=False)
    model.summary()

    # pull preprocessor
    preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset(model_id)
    tokenizer = preprocessor.tokenizer

    if type == type_ALL:
        print('training all layers')

        # run queries on DB to pull training data
        print('pulling data from DB')
        real_user_ratings_query = """SELECT a.text, au.user_id, au.user_rating
FROM articles a
JOIN articleuser au ON au.article_id = a.id
WHERE
  a.text != '' AND
  ((au.ai_rating IS NULL) OR (au.ai_rating <> 1.5)) AND
  (au.user_rating IS NOT NULL) 
ORDER BY
  COALESCE(ABS(2*au.user_rating-1),0) DESC, a.date DESC
LIMIT %s"""

        add_on_query = """SELECT a.text, a.date
FROM articles a
LEFT JOIN articleuser au ON au.article_id = a.id
WHERE
  a.text != '' AND au.ai_rating <> 1.5 AND au.user_rating IS NULL
GROUP BY a.text, a.date
ORDER BY a.date DESC, ABS(2*AVG(au.ai_rating)-1) DESC
LIMIT %s"""
        rating_texts = []
        rating_userids = []
        rating_ratings = []
        addon_texts = []
        with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
            cur = con.cursor()
            cur.execute(real_user_ratings_query, (max_user_ratings,))
            for row in cur:
                rating_texts.append(row[0])
                rating_userids.append(row[1])
                rating_ratings.append(row[2])

            cur.execute(add_on_query, (max_user_ratings - len(rating_ratings),))
            addon_texts = [row[0] for row in cur.fetchall()]

        print('pulled', len(rating_ratings), 'user ratings')

        # assemble data for length algorithm two-task training trick
        print('parsing lengths for', len(addon_texts), 'articles')
        lengths = tokenizer(addon_texts).row_lengths()
        loglengths = np.log(lengths+1)
        loglength_norm_layer = keras.layers.Normalization(axis=None)
        loglength_norm_layer.adapt(loglengths)
        print('Article log length mean:', loglength_norm_layer.mean.numpy()[0])
        print('Article log length sd:', loglength_norm_layer.variance.numpy()[0]**0.5)

        # convert raw data to numpy arrays 
        rating_texts = np.array(rating_texts)
        rating_userids = np.array(rating_userids)
        rating_ratings = np.array(rating_ratings)
        addon_texts = np.array(addon_texts)

        rec_count = rating_texts.shape[0]
        text_count = addon_texts.shape[0]
        print('recommendations:', rec_count)
        print('texts:', text_count)
        rec_val_cutoff = int(train_val_split * rec_count)
        text_val_cutoff = int(train_val_split * text_count)
        print('rec_val_cutoff:', rec_val_cutoff)
        print('text_val_cutoff:', text_val_cutoff)

        # create tf.data datasets
        recommend_text_ds = tf.data.Dataset.from_tensor_slices(rating_texts).batch(batch_size)
        recommend_user_ds = tf.data.Dataset.from_tensor_slices(rating_userids).batch(batch_size)
        recommend_ratings_ds = tf.data.Dataset.from_tensor_slices(rating_ratings).batch(batch_size)
        recommend_ds = tf.data.Dataset.zip(recommend_text_ds, recommend_user_ds, recommend_ratings_ds)
        recommend_ds = recommend_ds.map(lambda i, j, k: (preprocessor(i), j, k), num_parallel_calls=tf.data.AUTOTUNE)
        recommend_ds = recommend_ds.map(lambda i, j, k: ((i['token_ids'], i['padding_mask'], j), k))
        recommend_ds = recommend_ds.cache().prefetch(tf.data.AUTOTUNE)

        length_text_ds = tf.data.Dataset.from_tensor_slices(addon_texts).batch(batch_size)
        length_loglength_ds = tf.data.Dataset.from_tensor_slices(loglengths).batch(batch_size)
        length_ds = tf.data.Dataset.zip(length_text_ds, length_loglength_ds)
        length_ds = length_ds.map(lambda i, j: (preprocessor(i), loglength_norm_layer(j)), num_parallel_calls=tf.data.AUTOTUNE)
        length_ds = length_ds.map(lambda i, j: ((i['token_ids'], i['padding_mask']), j))
        length_ds = length_ds.cache().prefetch(tf.data.AUTOTUNE)

        val_recommend_ds = recommend_ds.take(int(rec_val_cutoff/batch_size)).repeat(int(text_val_cutoff/rec_val_cutoff)+1)
        train_recommend_ds = recommend_ds.skip(int(rec_val_cutoff/batch_size)).repeat(int((text_count-text_val_cutoff)/(rec_count-rec_val_cutoff))+1)
        val_length_ds = length_ds.take(int(text_val_cutoff/batch_size))
        train_length_ds = length_ds.skip(int(text_val_cutoff/batch_size))

        val_ds = tf.data.Dataset.zip(val_length_ds, val_recommend_ds)
        val_ds = val_ds.map(lambda i, j: (i[0]+j[0], (i[1], j[1])))
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

        train_ds = tf.data.Dataset.zip(train_length_ds, train_recommend_ds)
        train_ds = train_ds.map(lambda i, j: (i[0]+j[0], (i[1], j[1])))
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)

        for i, j in train_ds.as_numpy_iterator():
            print('train_ds input:', [(o.shape, o.dtype) for o in i])
            print('train_ds output:', [(o.shape, o.dtype) for o in j])
            break

        # compile model for output
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
            loss=['mse', 'mse'],
            metrics=['mse', 'mse'],
            loss_weights=[0.33, 0.67]
        )

        # determine pre-fine-tuning performance
        old_val_loss, old_val_length_out_mse, old_val_recommend_out_mse = model.evaluate(val_ds)
        print('un-fine-tuned val_loss:', old_val_loss)
        print('un-fine-tuned val_length_out_mse:', old_val_length_out_mse)
        print('un-fine-tuned val_recommend_out_mse:', old_val_recommend_out_mse)

        # train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=type_all_epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_recommend_out_mse', mode='min', patience=recommend_mse_patience, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=2, min_lr=1e-6, min_delta=0.0001),
                keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * (1.0 if epoch < 10 or lr < 1e-6 else 0.9)),
            ],
        )

        # determine post-fine-tuning performance
        new_val_loss, new_val_length_out_mse, new_val_recommend_out_mse = model.evaluate(val_ds)
        new_train_loss, new_train_length_out_mse, new_train_recommend_out_mse = model.evaluate(train_ds)
        print('fine-tuned val_loss:', new_val_loss)
        print('fine-tuned val_length_out_mse:', new_val_length_out_mse)
        print('fine-tuned val_recommend_out_mse:', new_val_recommend_out_mse)
        print('fine-tuned train_loss:', new_train_loss)
        print('fine-tuned train_length_out_mse:', new_train_length_out_mse)
        print('fine-tuned train_recommend_out_mse:', new_train_recommend_out_mse)

        # save model
        datestr = datetime.datetime.now().strftime('%Y%m%d%H%M')
        min_index = np.argmin(history.history['val_recommend_out_mse'])
        val_loss = "{loss:0.4f}".format(loss=history.history['val_recommend_out_mse'][min_index])
        savefile = 'newsreader_roberta_length-'+datestr+'-loss-'+val_loss+'.keras'
        model.compile( # reset optimizer
            jit_compile=True
        )
        model.save('/data/model/'+savefile)
        vol.commit()

        # if more than 10 epochs and fine-tuning made an improvement, crown a new best model
        if len(history.history['val_recommend_out_mse']) >= 10 and new_val_recommend_out_mse < old_val_recommend_out_mse and new_val_loss < old_val_loss:
            print('new best model at', '/data/model/'+savefile)
            with open('/data/model/best_model.txt', 'w') as f:
                f.write(savefile)
            print('trigger postTrain')
            postTrain.spawn() # initiate postTraining activities
        else:
            print('keeping current best model', model_path)

    else:
        print('Unrecognized training type:', type)

# function that coordinates post-training activities if a new model is crowned
# runs inference on newer content / previously higher rated content that are still unread
# also updates user embeddings and user histories
@app.function(timeout=1800, volumes={"/data": vol}, secrets=[modal.Secret.from_name("newsreader_psycopg2")])
def postTrain():
    import os
    import psycopg2 
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
    import json
    
    users_query = "SELECT id FROM users"
    article_query = """SELECT a.id, a.text
FROM articles a LEFT JOIN articleuser au ON au.article_id = a.id
WHERE (au.user_read IS NULL OR au.user_read = FALSE) AND a.text != '' AND au.ai_rating <> 1.5
GROUP BY a.id, a.text
ORDER BY COALESCE(MAX(au.ai_rating),0) DESC, MIN(au.rating_timestamp) ASC LIMIT %s
"""
    update_rating_query = "UPDATE articleuser SET ai_rating = %s, updated_at = NOW() WHERE article_id = %s AND user_id = %s"
    update_embedding_query = "UPDATE articles SET embedding = %s, updated_at = NOW() WHERE id = %s"
    get_user_history_query = """SELECT AVG(b.embedding)
FROM (
    SELECT a.embedding
    FROM articles a 
    JOIN articleuser au ON au.article_id = a.id
    JOIN users u ON au.user_id = u.id
    WHERE au.user_read = TRUE AND u.id = %s
    ORDER BY au.read_timestamp DESC 
    LIMIT %s
) b
"""
    update_user_query = "UPDATE users SET embedding = %s, recent_articles_read = %s, updated_at = NOW() WHERE id = %s"

    # get relevant parameters
    with open('/data/model/train_params.json', 'r') as f:
        trainparams = json.load(f)
    
    num_articles_posttrain = trainparams['num_articles_posttrain']
    num_articles_embedding_avg_posttrain = trainparams['num_articles_embedding_avg_posttrain']

    # get users list
    print('getting users list')
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor()
        cur.execute(users_query)
        users = [row[0] for row in cur.fetchall()]
    print(len(users), 'users pulled')

    # pull recent & likely-to-be recommended articles
    print('pulling', num_articles_posttrain, 'likely articles')
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor()
        cur.execute(article_query, (num_articles_posttrain,))
        results = cur.fetchall()
        article_ids = [tup[0] for tup in results]
        article_texts = [tup[1] for tup in results]
    
    # get recommendations and article embeddings
    print('rating', len(article_ids), 'articles')
    obj = modal.Cls.lookup("newsreader-recommend", "NewsreaderRecommendation")()
    ratings_array, text_embs = obj.rateTextUsers.remote(article_texts, users)

    # get new user embeddings
    print('getting user embeddings')
    userEmbeddings = obj.extractUserEmbeddings.remote()

    # insert article embeddings
    print('inserting article embeddings')
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        register_vector(con)
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        params = [(text_emb, article_id) for article_id, text_emb in zip(article_ids, text_embs)]
        try:
            psycopg2.extras.execute_batch(cur, update_embedding_query, params)
        except Exception as e:
            print('could not insert article embeddings with execute_batch:', e)
            cur.execute("ROLLBACK")
            return None
        else:
            con.commit()
            print('inserted', len(params), 'article embeddings')
    
    # update articleusers
    print('inserting articleuser ratings')
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        register_vector(con)
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        params = []
        for user_id, ratings in zip(users, ratings_array):
            for article_id, rating in zip(article_ids, ratings): 
                params.append((float(rating[0]), article_id, user_id))
        try:
            psycopg2.extras.execute_batch(cur, update_rating_query, params)
        except Exception as e:
            print('could not insert articleusers with execute_batch:', e)
            cur.execute("ROLLBACK")
            return None
        else:
            print('inserted', len(params), 'articleuser ratings')
            con.commit()
    
    # get new user history and update users
    print('getting user histories and updating user embeddings & history embeddings')
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        register_vector(con)
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        for user_id in users:
            cur.execute(get_user_history_query, (user_id, num_articles_embedding_avg_posttrain))
            history_vector = cur.fetchone()[0]
            cur.execute(update_user_query, (userEmbeddings[user_id], history_vector, user_id))
            con.commit()
    
    print('completed posttrain')