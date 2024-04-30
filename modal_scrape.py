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
    .pip_install('anthropic')
)

app = modal.App('newsreader-scrape', image=image)
vol = modal.Volume.from_name("newsreader-data")
scrape_queue = modal.Queue.from_name("newsreader-scrape-queue", create_if_missing=True)

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
    url_lls = list(scrapeSource.map(siteIndices))

    # flatten list and filter only for things that haven't been crawled before
    # update crawl record
    urls = [url for l in url_lls for url in l if url not in crawl_record]
    print('identified', len(urls), 'new urls for scraper')
    crawl_record = crawl_record + urls 

    # update crawl record up to maximum buffer
    max_records = 10000
    with open('/data/scrape/scraper_record.pickle', 'wb') as f:
        if len(crawl_record) > max_records:
            pickle.dump(crawl_record[-max_records:], f)
        else:
            pickle.dump(crawl_record, f)
        vol.commit()

    # scrape every URL in list and push to queue
    _ = list(scrapeUrl.starmap(zip(urls, [scrape_queue for url in urls])))

    # log articles in queue bite_size at a time, in parallel as needed
    bite_size = 200
    while scrape_queue.len() > 0:
        article_contexts = scrape_queue.get_many(bite_size, block=False)
        print('running logArticles on', len(article_contexts), 'and seeing', scrape_queue.len(), 'left in queue')
        logArticles.spawn(article_contexts, users)

# gets list of articles from a source
# only showing examples for Google Blog (using Google News as source) and Huggingface Blog
# would need to customize for sites of interest
@app.function()
def scrapeSource(sourceDict):
    import httpx
    import warnings 
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning) # ignore lxml warning
    max_urls = 100
    sourceType = sourceDict['type']
    sourceUrl = sourceDict['url']

    if sourceType == 'google_news':
        r = httpx.get(sourceUrl)
        soup = BeautifulSoup(r.content, 'lxml')
        list_of_urls = [loc.text for loc in soup.find_all('loc')[:max_urls]]
    
    elif sourceType == 'huggingface_rss':
        r = httpx.get(sourceUrl)
        soup = BeautifulSoup(r.content, 'xml')
        list_of_urls = [item.find('guid').text for item in soup.find_all('item')[:max_urls]]

    return list_of_urls

# scrapes a URL based on site type, pushes context to queue for logArticles to parse
# only showing examples for Google Blog and Huggingface Blog, would need to customize for sites of interest
@app.function()
def scrapeUrl(url, queue):
    import httpx 
    from bs4 import BeautifulSoup
    import datetime 
    import json

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

            title_tag = soup.find(lambda tag: tag.name=='meta' and 'property' in tag.attrs and 'content' in tag.attrs and tag['property']=='og:title')
            if title_tag:
                title = title_tag['content']
            else:
                title = None
            
            metatag = soup.find(lambda tag: tag.name == 'meta' and 'property' in tag.attrs and tag['property'] == 'article:modified_time')
            if metatag:
                datestr = metatag['content'][:10]
                dt_obj = datetime.datetime.strptime(datestr, '%Y-%m-%d')
                date_obj = datetime.date(dt_obj.year, dt_obj.month, dt_obj.day)
            elif soup.find(lambda tag: tag.name == 'meta' and 'property' in tag.attrs and tag['property'] == 'article:published_time'):
                metatag = soup.find(lambda tag: tag.name == 'meta' and 'property' in tag.attrs and tag['property'] == 'article:published_time')
                datestr = metatag['content'][:10]
                dt_obj = datetime.datetime.strptime(datestr, '%Y-%m-%d')
                date_obj = datetime.date(dt_obj.year, dt_obj.month, dt_obj.day)
            else:
                date_obj = None

            queue.put((article_text, title, author_name, author_href, date_obj, 'Google Blog', None, url))
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

            title_tag = soup.find(lambda tag: tag.name=='meta' and 'property' in tag.attrs and 'content' in tag.attrs and tag['property']=='og:title')
            if title_tag:
                title = title_tag['content']
            else:
                title = None
            
            date_tag = soup.find(lambda tag: tag.name=='span' and tag.get_text()[0:9]=='Published')
            if date_tag:
                dt_obj = datetime.datetime.strptime(date_tag.get_text().strip()[9:], '%B %d, %Y')
                date_obj = datetime.date(dt_obj.year, dt_obj.month, dt_obj.day)
            else:
                date_obj = None
            
            queue.put((article_text, title, author_name, author_href, date_obj, 'Canary Media', None, url))
            return True

        else:
            print('unsupported URL', url)
            return False
    except Exception as e:
        print('Issue with', url, 'Exception flagged:', e)
        return False

# invokes Anthropic to generate summaries
@app.function(secrets=[modal.Secret.from_name("newsreader_anthropic_key")], retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=60.0))
def summarize(article):
    import anthropic
    from bs4 import BeautifulSoup

    client = anthropic.Anthropic()
    try:
        # invoke Anthropic, truncate text if longer than 14,000 characters
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role":"user",
                    "content":[
                        {
                            "type": "text",
                            "text": "<CONTENT>\n" + (article if len(article) <= 14000 else "<TRUNCATED FOR LENGTH>" + article[:14000] + "</TRUNCATED FOR LENGTH>") + "\n</CONTENT>"
                        }
                    ]
                }
            ]
        )
        try:
            soup = BeautifulSoup(message.content[0].text, 'lxml')
            response = "<strong>Topics:</strong> " + ', '.join([topic.get_text(strip=True) for topic in soup.topics.find_all('topic')]) + "<br /><strong>Summary:</strong><br />" + soup.summary.get_text(strip=True)
            return response
        except Exception as e:
            print('failed to capture summary from raw XML:', message.content[0].text)
            return None
    except Exception as e:
        print('Anthropic issue, returning None:', e)
        return None

# reads outputs from scraper in parallel and runs rating algorithm on users
@app.function(timeout=1200, secrets=[modal.Secret.from_name("newsreader_psycopg2")])
def logArticles(article_contexts, users):
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
    import datetime
    import numpy as np

    # utility function to transform scraper outputs for database consumption
    def transform(ind_article):
        article_text, title, author_name, author_href, date_obj, source, url, summary, embedding = ind_article 
        return article_text, title, author_name if author_name else 'Author', author_href if author_href else '', date_obj, source, url, summary, embedding

    # invoke modal_recommend class
    obj = modal.Cls.lookup("newsreader-recommend", "NewsreaderRecommendation")()

    articles = [article_context[0] for article_context in article_contexts]

    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        register_vector(con) # allow Python connection to support numpy arrays natively
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        if articles:
            article_contexts = [article_context[0:6]+(article_context[7],) for article_context in article_contexts]
            summaries = summarize.map(articles) # summarize articles with Anthropic in parallel

            # run rating algorithm
            user_ratings, text_embeddings = obj.rateTextUsers.remote(articles, users) 
            article_contexts = [article_context+(summary, text_embedding,) for article_context, summary, text_embedding in zip(article_contexts, summaries, text_embeddings)]
            article_params = list(map(transform, article_contexts))

            # insert articles into database
            try:
                article_ids = psycopg2.extras.execute_values(cur, """INSERT INTO articles (text, title, author_name, author_href, date, source, url, summary, embedding) VALUES %s RETURNING id""", article_params, fetch=True)
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
                article_user_params = [(user_id, False, float(user_article_rating[0]), datetime.datetime.utcnow(), article_id) for user_article_rating, article_id in zip(user_rating, article_ids)]
                try:
                    article_user_ids = psycopg2.extras.execute_values(cur, """INSERT INTO articleuser (user_id, user_read, ai_rating, rating_timestamp, article_id) VALUES %s RETURNING id""", article_user_params, fetch=True)
                    con.commit()
                    count += len(article_user_ids)
                except:
                    cur.execute("ROLLBACK")
                    for user_id, user_article_rating, article_id in article_user_params:
                        try:
                            cur.execute("""INSERT INTO articleuser (user_id, user_read, ai_rating, rating_timestamp, article_id) VALUES (%s, FALSE, %s, NOW(), %s)""", (user_id, user_article_rating, article_id))
                            con.commit()
                            count += 1
                        except Exception as e:
                            cur.execute("ROLLBACK")
                            print('Could not insert rating of article', article_id, 'for user', user_id, 'due to psycopg2 error:', e)
            
            print('successfully added', count, 'articleuser rows')

    return True 
