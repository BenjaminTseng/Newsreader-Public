import modal

jax_image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('jax[cuda12]==0.4.35', extra_options="-U")
    .pip_install('keras==3.6')
    .pip_install('keras-hub==0.17')
    .pip_install('psycopg2-binary')
    .pip_install('pgvector')
    .env({"KERAS_BACKEND":"jax"})
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION":"1.0"})
)

app = modal.App("newsreader-train", image=jax_image)
vol = modal.Volume.from_name("newsreader-data")

# process run twice a month
# type currently unnecessary variable but can be used in the future to change training types
@app.function(gpu="L4", timeout=10800, volumes={"/data": vol}, secrets=[modal.Secret.from_name("newsreader_psycopg2")], schedule=modal.Cron("30 9 14,28 * *"))
def train():
    # eliminate always there tensorflow warnings from the logs
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import warnings 
    warnings.filterwarnings('ignore')

    import keras
    import keras_hub
    import jax
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
    rating_out_mse_patience = trainparams['rating_out_mse_patience']

    # path for model
    with open('/data/model/best_model.txt', 'r') as f:
        model_path = '/data/model/'+f.read().strip()

    # pull model
    print('loading model', model_path)
    model = keras.models.load_model(model_path, compile=False)
    model.summary()

    # pull preprocessor
    preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(model_id)

    # run queries on DB to pull training data
    print('pulling data from DB')
    real_user_ratings_query = """SELECT
  a.text,
  au.user_id,
  au.user_rating,
  a.token_length
FROM articles a
JOIN articleuser au ON au.article_id = a.id
WHERE
  a.text != '' AND
  ((au.ai_rating IS NULL) OR (au.ai_rating <> 1.5)) AND
  (au.user_rating IS NOT NULL)
ORDER BY
  a.date DESC, COALESCE(ABS(2*au.user_rating-1),0) DESC
LIMIT %s"""
    
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor()
        cur.execute(real_user_ratings_query, (max_user_ratings,))
        real_user_ratings = []
        for r in cur:
            real_user_ratings.append(r)

    print(len(rating_ratings), 'user ratings')
    rec_count = len(real_user_ratings)
    rec_val_cutoff = int(train_val_split * rec_count)
    print('rec_val_cutoff:', rec_val_cutoff)

    all_new_cutoff = int(rec_val_cutoff/2)
    print('all_new_cutoff:', all_new_cutoff)
    all_new = real_user_ratings[:all_new_cutoff]
    mix = real_user_ratings[all_new_cutoff:]
    np.random.shuffle(mix)
    np.random.shuffle(all_new)
    real_user_ratings = all_new + mix

    # assemble data for length algorithm two-task training trick
    lengths = np.array([r[3] for r in real_user_ratings])
    loglengths = np.log(lengths+1)
    loglength_norm_layer = keras.layers.Normalization(axis=None)
    loglength_norm_layer.adapt(loglengths)
    print('Article log length mean:', loglength_norm_layer.mean)
    print('Article log length sd:', keras.ops.sqrt(loglength_norm_layer.variance))

    # convert raw data to numpy arrays 
    rating_texts = [r[0] for r in real_user_ratings]
    rating_userids = [r[1] for r in real_user_ratings]
    rating_ratings = [r[2] for r in real_user_ratings]
    rating_texts = np.array(rating_texts)
    rating_userids = np.array(rating_userids)
    rating_ratings = np.array(rating_ratings)

    # create tf.data datasets
    text_ds = tf.data.Dataset.from_tensor_slices(rating_texts).batch(batch_size)
    user_ds = tf.data.Dataset.from_tensor_slices(rating_userids).batch(batch_size)
    ratings_ds = tf.data.Dataset.from_tensor_slices(rating_ratings).batch(batch_size)
    lengths_ds = tf.data.Dataset.from_tensor_slices(lengths).batch(batch_size)
    ds = tf.data.Dataset.zip(text_ds, user_ds, ratings_ds, lengths_ds)
    ds = ds.map(lambda i, j, k, l: (preprocessor(i), j, 2*k-1, loglength_norm_layer(tf.math.log(tf.cast(l, dtype=tf.float32)+1))), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda i, j, k, l: ((i['token_ids'], i['padding_mask'], j), (l, k)), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)

    val_ds = ds.take(int(rec_val_cutoff/batch_size))
    train_ds = ds.skip(int(rec_val_cutoff/batch_size))

    for i, j in train_ds.as_numpy_iterator():
        print('train_ds input:', [(o.shape, o.dtype) for o in i])
        print('train_ds output:', [(o.shape, o.dtype) for o in j])
        break

    # compile model for output
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
        loss=['mse', 'mse'],
        metrics=['mse', 'mse'],
        loss_weights=[0.33, 0.67],
        jit_compile=True # take advantage of JAX heavy use of XLA
    )

    # determine pre-fine-tuning performance
    old_val_loss, old_val_length_out_mse, old_val_rating_out_mse, _, _ = model.evaluate(val_ds)
    print('un-fine-tuned val_loss:', old_val_loss)
    print('un-fine-tuned val_length_out_mse:', old_val_length_out_mse)
    print('un-fine-tuned val_rating_out_mse:', old_val_rating_out_mse)

    # train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=type_all_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_rating_out_mse', mode='min', patience=rating_out_mse_patience, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=2, min_lr=2e-6, min_delta=0.0001, verbose=1),
            keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * (1.0 if epoch < 10 or lr < 1e-6 else 0.9), verbose=1),
        ],
    )

    # determine post-fine-tuning performance
    new_val_loss, new_val_length_out_mse, new_val_rating_out_mse, _, _ = model.evaluate(val_ds)
    new_train_loss, new_train_length_out_mse, new_train_rating_out_mse, _, _ = model.evaluate(train_ds)
    print('fine-tuned val_loss:', new_val_loss)
    print('fine-tuned val_length_out_mse:', new_val_length_out_mse)
    print('fine-tuned val_rating_out_mse:', new_val_rating_out_mse)
    print('fine-tuned train_loss:', new_train_loss)
    print('fine-tuned train_length_out_mse:', new_train_length_out_mse)
    print('fine-tuned train_rating_out_mse:', new_train_rating_out_mse)

    # save model
    datestr = datetime.datetime.now().strftime('%Y%m%d%H%M')
    min_index = np.argmin(history.history['val_rating_out_mse'])
    val_loss = "{loss:0.4f}".format(loss=history.history['val_rating_out_mse'][min_index])
    savefile = 'iggregate_onlyratings_dot-'+datestr+'-loss-'+val_loss+'.keras'
    model.compile( # reset optimizer
        jit_compile=True
    )
    model.save('/data/model/'+savefile)
    vol.commit()

    # if more than 10 epochs and fine-tuning made an improvement, crown a new best model
    if len(history.history['val_rating_out_mse']) > (rating_out_mse_patience+1) and new_val_rating_out_mse < old_val_rating_out_mse and new_val_loss < old_val_loss:
        print('new best model at', '/data/model/'+savefile)
        with open('/data/model/best_model.txt', 'w') as f:
            f.write(savefile)
        vol.commit()
        print('trigger postTrain')
        postTrain.spawn() # initiate postTraining activities
    else:
        print('keeping current best model', model_path)


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
    article_query = """(SELECT a.id, a.text
FROM articles a LEFT JOIN articleuser au ON au.article_id = a.id
WHERE (au.id IS NULL) OR ((au.user_read IS NULL OR au.user_read = FALSE) AND a.text != '' AND au.ai_rating <> 1.5)
GROUP BY a.id, a.text
ORDER BY COALESCE(MAX(au.ai_rating),0) DESC, MIN(au.rating_timestamp) ASC LIMIT %s)
UNION
(SELECT a.id, a.text
FROM articles a LEFT JOIN articleuser au ON au.article_id = a.id
WHERE (au.id IS NULL) OR ((au.user_read IS NULL OR au.user_read = FALSE) AND a.text != '' AND au.ai_rating <> 1.5)
GROUP BY a.id, a.text
ORDER BY a.updated_at DESC LIMIT %s)
"""
    update_rating_query = "UPDATE articleuser SET ai_rating = %s, rating_timestamp = NOW(), updated_at = NOW() WHERE article_id = %s AND user_id = %s"
    update_embedding_query = "UPDATE articles SET embedding = %s, updated_at = NOW() WHERE id = %s"
    get_user_history_query = """SELECT a.embedding
FROM articles a 
JOIN articleuser au ON au.article_id = a.id
JOIN users u ON au.user_id = u.id
WHERE au.user_read = TRUE AND u.id = %s
ORDER BY au.read_timestamp DESC 
LIMIT %s"""
    update_user_query = "UPDATE users SET embedding = %s, recent_articles_read = %s, updated_at = NOW() WHERE id = %s"

    # get relevant parameters
    with open('/data/model/train_params.json', 'r') as f:
        trainparams = json.load(f)
    
    with open('/data/api/fetchparams.json', 'r') as f:
        fetchparams = json.load(f)
    
    num_articles_posttrain = trainparams['num_articles_posttrain']
    user_history_ema = trainparams['user_history_ema']
    article_history_inertia = fetchparams['article_history_inertia']

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
    ratings_array, text_embs, _ = obj.rateTextUsers.remote(article_texts, users)

    # get new user embeddings
    print('getting user embeddings')
    user_embs = obj.embedUser.remote(users)

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
    
    # get new user history and update users
    print('getting user histories and updating user embeddings & history embeddings')
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        register_vector(con)
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        for user_id, user_emb in zip(users, user_embs):
            cur.execute(get_user_history_query, (user_id, user_history_ema))
            first = True
            for row in cur:
                if first:
                    history_vector = row[0]
                    first = False
                else:
                    history_vector = history_vector*article_history_inertia+row[0]*(1-article_history_inertia)

            cur.execute(update_user_query, (user_emb, history_vector, user_id))
        con.commit()

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
    
    print("updating fetch_ratings and clearing out articleusers with missing similarity scores")
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        try:
            cur.execute("""UPDATE articleuser
SET article_user_similarity = 0.025*(a.embedding <-> u.recent_articles_read)
FROM articles a, users u
WHERE articleuser.article_id = a.id
AND articleuser.user_id = u.id
AND articleuser.article_user_similarity IS NULL;""")
                
            cur.execute("""UPDATE articleuser
SET fetch_rating = CASE 
    WHEN su.always_show = TRUE THEN 100.0 
    ELSE (
        %s * EXP(GREATEST((a.date - CURRENT_DATE)::INT, %s) / %s) + 
        %s * COALESCE(articleuser.user_rating, articleuser.ai_rating) + 
        %s * articleuser.article_user_similarity
    ) 
END
FROM articles a, sourceuser su
WHERE articleuser.article_id = a.id
AND su.user_id = articleuser.user_id
AND su.source_id = a.source;""", (fetchparams['recency_factor'], 
                                    fetchparams['day_decay_threshold'],
                                    fetchparams['day_decay_scale'],
                                    fetchparams['rating_factor'],
                                    fetchparams['similarity_factor'],
                                ))
            con.commit()
        except:
            cur.execute("ROLLBACK")
            print('Could not update fetch_ratings due to psycopg2 error:', e)
            return False

    print('completed posttrain')