import modal
import os

# Modal
image = (
    modal.Image.debian_slim(python_version='3.10')
    .pip_install('httpx')
    .pip_install('beautifulsoup4')
    .pip_install('lxml')
    .pip_install('psycopg2-binary')
    .pip_install('pgvector')
#    .pip_install('anthropic')
    .pip_install('google-generativeai')
)

app = modal.App('newsreader-scrape', image=image)
vol = modal.Volume.from_name("newsreader-data")

# Anthropic system prompt
system_prompt = """You will be provided with text content scraped from a website in between <CONTENT> and </CONTENT>. The scraper replaces new paragraphs and <divs> with "<|>". Because the scraper only looks for text, interactive elements and images will be omitted. Because no scraper is 100% reliable, some text and sidebars / figures may also be omitted. The content that will be provided has already passed a basic relevancy / rating filter, so you can assume the content is of relatively higher quality.

Carefully read the text, summarize the key points, and explain why the reader may wish to read the entire piece in the following XML format:

<RESPONSE>
    <SUMMARY><up to 100 word summary of the piece in complete sentences></SUMMARY>
    <TOPICS>
        <TOPIC><3-10 topics that can help categorize this piece within a broader dataset of general interest articles, at least 1 should be broad (i.e. \"Technology\", \"Pharmaceuticals\") and at least 2 should be specific></TOPIC>
    </TOPICS>

ONLY return this XML and NOTHING else so that the response can be parsed easily by loading the response into Python's BeautifulSoup"""

# headers to pass
# you may want to experiment with user-agent strings here to get better performance
headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh-TW;q=0.7,zh;q=0.6',
    'cache-control': 'max-age=0',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'cross-site',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1'
}

# coordinates the scrape process by pulling the sources -> triggering page scrapes -> logging in DB
# runs every few hours every day 
@app.function(timeout=1200, volumes={"/data": vol}, schedule=modal.Cron("15 3,13,16,19,22 * * *"), secrets=[modal.Secret.from_name("newsreader_psycopg2")])
def scrapeProcess():
    import psycopg2
    import psycopg2.extras
    import pickle
    import json 

    # get all users
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT id FROM users")
        users = [row[0] for row in cur.fetchall()]

    # pull sites to crawl from JSON file
    # each site includes a name, type, and URL
    with open('/data/scrape/siteindices.json', 'r') as f:
        siteIndices = json.load(f)
    
    # pull current crawl record but create it from database if it doesn't exist / error
    try:
        with open('/data/scrape/scraper_record.pickle', 'rb') as f:
            crawl_record = pickle.load(f)
    except:
        # use totality of URLs loaded if pickle fails
        with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
            cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute("SELECT url FROM articles ORDER BY id ASC")
            crawl_record = [row['url'] for row in cur]

    # scrape all the sources in parallel for URLs
    scrape_data = list(scrapeSource.map(siteIndices))
    scrape_data = [list_item for ll in scrape_data for list_item in ll ]

    # flatten list and filter only for things that haven't been crawled before
    # update crawl record
    scrape_data = [sd for sd in scrape_data if sd[1] not in crawl_record]
    print('identified', len(scrape_data), 'new urls for scraper')

    with modal.Queue.ephemeral() as scrape_queue:
        # scrape every URL in list and push to queue
        scrape_output = list(scrapeUrl.starmap([(scrape_queue,) + sd for sd in scrape_data]))
        crawl_record += [scrape_data[i][1] for i in range(len(scrape_output)) if scrape_output[i] and scrape_output[i] != -1]

        # log articles in queue bite_size at a time, in parallel as needed
        bite_size = 200
        while scrape_queue.len() > 0:
            article_contexts = scrape_queue.get_many(bite_size, block=False)
            print('running logArticles on', len(article_contexts), 'and seeing', scrape_queue.len(), 'left in queue')
            logArticles.spawn(article_contexts, users)

    # update crawl record up to maximum buffer
    max_records = 10000
    with open('/data/scrape/scraper_record.pickle', 'wb') as f:
        if len(crawl_record) > max_records:
            pickle.dump(crawl_record[-max_records:], f)
        else:
            pickle.dump(crawl_record, f)
        vol.commit()

# gets list of articles from a source
# only showing examples for Google Blog (using Google News as source) and Huggingface Blog
# would need to customize for sites of interest
@app.function()
def scrapeSource(sourceDict):
    import httpx
    import warnings 
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    import html 
    import datetime

    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning) # ignore lxml warning
    max_urls = 100
    sourceType = sourceDict['type']
    sourceUrl = sourceDict['url']
    sourceId = sourceDict['source_id']
    sourceSite = sourceDict['site']
    scrape_data = []

    if sourceType == 'google_news':
        try:
            r = httpx.get(sourceUrl, headers=headers, timeout=10.0)
        except Exception as e:
            print('Issue with scraping', sourceUrl, 'exception:', e)
        else:
            soup = BeautifulSoup(r.content, 'xml')
            for loc in soup.find_all('url')[:max_urls]:
                title = None
                article_text = None 
                author_name = None
                author_href = None
                date_obj = None

                url = loc.find('loc').get_text().strip()
                title = html.unescape(loc.find('news:title').get_text().strip())
                try:
                    datestr = loc.find('news:publication_date').get_text()
                    dt_obj = datetime.datetime.strptime(datestr[:10], '%Y-%m-%d')
                    date_obj = datetime.date(dt_obj.year, dt_obj.month, dt_obj.day)
                except: 
                    date_obj = None
                scrape_data.append((sourceId, url, article_text, title, author_name, author_href, date_obj))
    
    elif sourceType == 'huggingface_rss':
        try:
            r = httpx.get(sourceUrl, headers=headers, timeout=10.0)
        except Exception as e:
            print('Issue with scraping', sourceUrl, 'exception:', e)
        else:
            soup = BeautifulSoup(r.content, 'xml')
            for loc in soup.find_all('item')[:max_urls]:
                title = loc.find('title').get_text()
                article_text = None 
                author_name = None 
                author_href = None 
                datestr = loc.find('pubDate').get_text()
                date_obj = datetime.datetime.strptime(datestr, '%a, %d %b %Y %H:%M:%S %Z')
                scrape_data.append((sourceId, url, article_text, title, author_name, author_href, date_obj))

    return scrape_data

# scrapes a URL based on site type, pushes context to queue for logArticles to parse
# only showing examples for Google Blog and Huggingface Blog, would need to customize for sites of interest
@app.function()
def scrapeUrl(queue, sourceId, url, article_text, title, author_name, author_href, date_obj):
    import httpx 
    from bs4 import BeautifulSoup
    import datetime 
    import json
    import html

    # utility function to deal with tricky unicode characters
    def remove_unicode(s):
        s = s.replace(u'‘', "'")
        s = s.replace(u'’', "'")
        s = s.replace(u'“', '"')
        s = s.replace(u'”', '"')
        s = s.replace(u'–', '-')
        s = s.replace(u'—', '-')
        s = s.replace(u'\xa0', ' ')
        return s

    try:
        if 'https://blog.google/' in url:
            r = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
            soup = BeautifulSoup(r.content, 'lxml')

            article_tag = soup.find('section', class_='article-container')
            if article_tag:
                for tag in article_tag.find_all(lambda tag: tag.name[0] == 'h' or tag.name == 'p' or tag.name == 'li'):
                    if tag.get_text(strip=True):
                        tag.insert_after(' <|> ')
                article_text = remove_unicode(article_tag.get_text())
                article_text = ' '.join(article_text.split())
            else:
                print('failed to parse', url, 'no article-container detected')
                return False

            byline_tag = soup.find(lambda tag: tag.name=='meta' and 'name' in tag.attrs and 'content' in tag.attrs and tag['name']=='article-author')
            author_name = byline_tag.get_text()
            author_href = None # Google Blog doesn't link to authors

            queue.put((sourceId, url, article_text, title, author_name, author_href, date_obj))
            return True

        elif 'https://huggingface.co/blog/' in url:
            r = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
            soup = BeautifulSoup(r.content, 'lxml')

            article_tag = soup.find('div', class_='blog-content')
            if article_tag:
                for tag in article_tag.find_all(lambda tag: tag.name=='div' and 'class' in tag.attrs and ('not-prose' in tag['class'] or 'mb-4' in tag['class'])):
                    tag.decompose()
                
                for tag in article_tag.find_all(lambda tag: tag.name[0] == 'h' or tag.name == 'p' or tag.name == 'li'):
                    if tag.get_text(strip=True):
                        tag.insert_after(' <|> ')
                article_text = remove_unicode(article_tag.get_text())
                article_text = ' '.join(article_text.split())
            else:
                print('failed to parse', url, 'no article-container detected')
                return False
            
            # for simplicity leaving this out
            author_href = None
            author_name = None
            
            queue.put((article_text, title, author_name, author_href, date_obj, 'Canary Media', None, url))
            return True

        else:
            print('unsupported URL', url)
            return False
    except Exception as e:
        print('Issue with', url, 'Exception flagged:', e)
        return False

# invokes Google Gemini to generate summaries
@app.function(secrets=[modal.Secret.from_name("newsreader_google_ai_key")], retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=60.0))
def summarize(article):
    import google.generativeai as genai
    import os
    from bs4 import BeautifulSoup

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gen_config = genai.types.GenerationConfig(max_output_tokens=1000, temperature=0.0)
    model = genai.GenerativeModel("gemini-1.5-flash-002", system_instruction=system_prompt)
    try:
        # invoke Gemini Flash, truncate text if longer than 100,000 characters
        message = model.generate_content(
            "<CONTENT>\n" + (article if len(article) <= 100000 else "<TRUNCATED FOR LENGTH>" + article[:100000] + "</TRUNCATED FOR LENGTH>") + "\n</CONTENT>",
            generation_config=gen_config
        )
        try:
            soup = BeautifulSoup(message.text, 'lxml')
            response = "<strong>Topics:</strong> " + ', '.join([topic.get_text(strip=True) for topic in soup.topics.find_all('topic')]) + "<br /><strong>Summary:</strong><br />" + soup.summary.get_text(strip=True)
            return response
        except Exception as e:
            print('failed to capture summary from raw XML:', message.content[0].text)
            return None
    except Exception as e:
        print('Google issue, returning None:', e)
        return None

# reads outputs from scraper in parallel and runs rating algorithm on users
@app.function(timeout=1200, secrets=[modal.Secret.from_name("newsreader_psycopg2")])
def logArticles(article_contexts, users):
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
    import datetime
    import numpy as np
    import json 

    # fetch relevant parameters
    with open('/data/api/fetchparams.json', 'r') as f:
        fetchparams = json.load(f)

    # utility function to transform scraper outputs for database consumption
    def transform(ind_article):
        sourceId, url, article_text, title, author_name, author_href, date_obj, summary, text_embedding, token_length = ind_article
        return sourceId, url, article_text, title, author_name if author_name else 'Author', author_href if author_href else '', date_obj, summary, text_embedding, token_length

    # invoke modal_recommend class
    obj = modal.Cls.lookup("newsreader-recommend", "NewsreaderRecommendation")()

    embedding_contexts = [article_context[0] for article_context in article_contexts]
    article_texts = [article_context[2] for article_context in embedding_contexts]

    # summarize articles with Anthropic in parallel
    summaries = summarize.map(article_texts) 

    # run rating algorithm
    user_ratings, text_embeddings, token_lengths = obj.rateTextUsers.remote(article_texts, users) 

    embedding_contexts = [article_context+(summary, text_embedding, token_length) for article_context, summary, text_embedding, token_length in zip(embedding_contexts, summaries, text_embeddings, token_lengths.tolist())]
    article_params = [transform(article_context) for article_context in embedding_contexts]

    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        register_vector(con) # allow Python connection to support numpy arrays natively
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # insert articles into database
        try:
            article_ids = psycopg2.extras.execute_values(cur, """INSERT INTO articles (source, url, text, title, author_name, author_href, date, summary, embedding, token_length) VALUES %s RETURNING id""", article_params, fetch=True)
            con.commit()
            print("successfully added", len(article_ids), "articles rows")
        except Exception as e:
            print('Ran into psycopg2 error while inserting', len(article_contexts), 'articles:', e)
            cur.execute("ROLLBACK")
            return False

        # insert AI ratings into database    
        count = 0
        article_ids = [article_id[0] for article_id in article_ids]
        for user_rating, user_id in zip(user_ratings, users):
            article_user_params = [(user_id, False, float(user_article_rating), datetime.datetime.utcnow(), article_id) for user_article_rating, article_id in zip(user_rating, article_ids)]
            try:
                article_user_ids = psycopg2.extras.execute_values(cur, """INSERT INTO articleuser (user_id, user_read, ai_rating, rating_timestamp, article_id) VALUES %s RETURNING id""", article_user_params, fetch=True)
                con.commit()
                count += len(article_user_ids)
            except:
                cur.execute("ROLLBACK")
                print('Could not insert', len(article_user_params), 'ai_ratings of articles due to psycopg2 error:', e)
                return False
        
        # insert fake AI ratings into database
        fake_article_user_params = []
        for article_id in article_ids[user_ratings[0].shape[0]:]:
            for user_id in users:
                fake_article_user_params.append((user_id, False, 0.5, datetime.datetime.utcnow(), article_id))
        
        try:
            article_user_ids = psycopg2.extras.execute_values(cur, """INSERT INTO articleuser (user_id, user_read, ai_rating, rating_timestamp, article_id) VALUES %s RETURNING id""", fake_article_user_params, fetch=True)
            con.commit()
            count += len(article_user_ids)
        except:
            cur.execute("ROLLBACK")
            print('Could not insert', len(fake_article_user_params), 'fake ai_ratings of articles due to psycopg2 error:', e)
            return False

        print('successfully added', count, 'articleuser rows')

        # update articleuser similarity and fetch_rating
        print("updating fetch_ratings and clearing out articleusers with missing similarity scores")
        try:
            cur.execute("""UPDATE articleuser
SET article_user_similarity = 1 - (a.embedding <=> u.recent_articles_read)
FROM articles a, users u
WHERE articleuser.article_id = a.id
AND articleuser.user_id = u.id
AND articleuser.article_user_similarity IS NULL;""")
                
            cur.execute("""UPDATE articleuser
SET fetch_rating = CASE 
    WHEN su.always_show = TRUE THEN 100.0 
    ELSE (
        %s * EXP(LEAST((a.date - CURRENT_DATE)::INT, %s) / %s) + 
        %s * COALESCE(articleuser.user_rating, articleuser.ai_rating) + 
        %s * articleuser.article_user_similarity
    ) 
END
FROM articles a, sourceuser su
WHERE articleuser.article_id = a.id
AND su.user_id = articleuser.user_id
AND su.source_id = a.source
AND articleuser.fetch_rating IS NULL;""", (fetchparams['recency_factor'], 
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

    return True 

@app.function(timeout=3600, volumes={"/data": vol}, schedule=modal.Cron("0 4 * * *"), secrets=[modal.Secret.from_name("iggregate_psycopg2")])
def cleanUp():
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
    import time
    import json

    print('running cleanUp')

    # get relevant parameters
    with open('/data/model/train_params.json', 'r') as f:
        trainparams = json.load(f)

    with open('/data/api/fetchparams.json', 'r') as f:
        fetchparams = json.load(f)
    
    clean_up_summaries_bite_size = trainparams['clean_up_summaries_bite_size']
    clean_up_summaries_max_run = trainparams['clean_up_summaries_max_run']

    # clean up ratings
    print('add ratings to articleusers without')
    add_ratings_to_null_query = """INSERT INTO articleuser (article_id, user_id, ai_rating, created_at, updated_at, rating_timestamp)
SELECT 
    a.id AS article_id, 
    u.id AS user_id, 
    0.5*(1-(a.embedding<=>u.embedding))+0.5 AS ai_rating,
    NOW() AS created_at,
    NOW() AS updated_at,
    NOW() AS rating_timestamp
FROM articles a
JOIN users u ON (1=1)
LEFT JOIN articleuser au ON a.id = au.article_id AND u.id = au.user_id
WHERE au.article_id IS NULL
"""
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        ids = cur.execute(add_ratings_to_null_query)
        if ids and len(ids) > 0:
            print(len(ids), 'missing articleusers added')
        else:
            print('no articles missing articleusers')
        con.commit()

    print('add ratings to articleusers with NULL ai_ratings')
    add_ratings_to_nullrating_query = """UPDATE articleuser
SET 
    ai_rating = 0.5 + 0.5 * (1 - (a.embedding <=> u.embedding)),
    rating_timestamp = NOW(),
    updated_at = NOW()
FROM 
    articles a, 
    users u
WHERE 
    articleuser.article_id = a.id
    AND articleuser.user_id = u.id
    AND (articleuser.ai_rating IS NULL OR articleuser.ai_rating > 1.5)
"""
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(add_ratings_to_nullrating_query)
        con.commit()

    # update all fetch_ratings due to dates and any where article similarity is NULL
    print("updating fetch_ratings and clearing out articleusers with missing similarity scores")
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        try:
            cur.execute("""UPDATE articleuser
SET article_user_similarity = 1 - (a.embedding <=> u.recent_articles_read)
FROM articles a, users u
WHERE articleuser.article_id = a.id
AND articleuser.user_id = u.id
AND articleuser.article_user_similarity IS NULL;""")
                
            cur.execute("""UPDATE articleuser
SET fetch_rating = CASE 
    WHEN su.always_show = TRUE THEN 100.0 
    ELSE (
        %s * EXP(LEAST((a.date - CURRENT_DATE)::INT, %s) / %s) + 
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
        except Exception as e:
            cur.execute("ROLLBACK")
            print('Could not update fetch_ratings due to psycopg2 error:', e)

    print("reindexing articleuser")
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor()
        cur.execute("REINDEX TABLE articleuser")