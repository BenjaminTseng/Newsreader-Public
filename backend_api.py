import modal
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, Form, Request, Cookie, Header, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import fastapi
import datetime
import os
from typing import List
from jose import JWTError, jwt
from passlib.context import CryptContext

root_path = '/data/api/'

# Modal image initialization
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('fastapi[standard]==0.115.5')
    .pip_install('pydantic==2.10')
    .pip_install('psycopg2-binary==2.9.10')
    .pip_install('pgvector==0.3.6')
    .pip_install('jinja2==3.1.4')
    .pip_install('passlib[bcrypt]')
    .pip_install('python-jose[cryptography]')
    .pip_install('python-multipart')
    .pip_install('supabase==2.10')
)

app = modal.App('newsreader', image=image)
vol = modal.Volume.from_name("newsreader-data")

# Fast API initiation
web_app = FastAPI()

# set up cryptographic context for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Data model for FastAPI POST parameters
class User(BaseModel):
    id: int
    email: str 
    hashed_password: str

class FeedItem(BaseModel):
    articleId: int
    articleUrl: str
    title: str 
    date: datetime.date 
    author_name: str
    author_href: str
    source: str 
    summary: str 
    score: float
    user_rating: str
    user_read: str

class FeedItemListResponse(BaseModel):
    offset: int
    items: List[FeedItem]

class MarkItemReadPayload(BaseModel):
    articleId: int
    status: bool | None = True

class RateItemPayload(BaseModel):
    articleId: int
    rating: float | None

# utility function for getting logged in user, uses Supabase for performance
def get_user(email: str):
    import os 
    import supabase

    try:
        client = supabase.create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
        response = client.table('users').select('id', 'email', 'hashed_password').eq('email',email).execute()
    except Exception as e:
        print('issue conecting to supabase in get_user, will return error:', e)
        return None
    else:
        if len(response.data) == 1:
            return User(id=response.data[0]['id'], email=email, hashed_password=response.data[0]['hashed_password'])
        else:
            return None

# for use with HTML serving, assumes access token is in a Cookie
def get_current_user_cookie(access_token: str | None = Cookie(None)):
    import os 
    if not access_token:
        return None

    try:
        payload = jwt.decode(access_token, os.environ['SECRET_KEY'], algorithms=["HS256"])
        email: str = payload.get("email")
        if email is None:
            return None
    except JWTError:
        return None
    return get_user(email)

# for use with API, assumes access token is in header as Bearer token
def get_current_user_bearer(authorization: str | None = Header(None)):
    import os 
    credentials_exception = HTTPException(
        status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not authorization or authorization[0:7] != 'Bearer ':
        raise credentials_exception

    try:
        payload = jwt.decode(authorization[7:], os.environ['SECRET_KEY'], algorithms=["HS256"])
        email: str = payload.get("email")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(email)
    if user is None:
        raise credentials_exception
    else:
        return user

# for use with fetch articles endpoints
def fetchCall(user_id: int, n: int, offset: int=0, unread: bool=True, bad: bool=False, random: bool=False, source: List[int] = Query(None)):
    import json 
    import psycopg2
    import psycopg2.extras

    # utility function to make response strings behave better
    def remove_quotes(s):
        s = s.replace(u'`', "'")
        s = s.replace(u'"', "'")
        s = s.replace(u'\xa0', ' ')
        s = s.replace("""<!DOCTYPE html PUBLIC '-//W3C//DTD HTML 4.0 Transitional//EN' 'http://www.w3.org/TR/REC-html40/loose.dtd'>""","")
        s = s.replace("\n", " ")
        return s
    
    # construct fetch query
    base_query = "SELECT\n"
    base_query += "    a.id, a.url, a.title, a.date, a.author_name, a.author_href, s.name AS source, s.id AS source_id, a.summary, au.ai_rating, au.user_rating, au.user_read\n"
    base_query += "FROM articles a\nJOIN articleuser au ON a.id = au.article_id\nJOIN users u ON u.id = au.user_id\nJOIN sources s ON a.source = s.id\n"
    base_query += "WHERE\n    au.user_id = %s\n    AND au.ai_rating IS NOT NULL "

    if unread:
        base_query += " AND au.user_read = FALSE"
    
    if source:
        source_tuple = str(tuple(source)+(-1,)) if len(source) == 1 else str(tuple(source))
        base_query += " AND a.source IN " + source_tuple
        source_name_query = "SELECT name FROM sources WHERE id IN " + source_tuple

    if random:
        base_query += "\nORDER BY RANDOM()\n"
    else:
        if bad:
            base_query += "\nORDER BY au.fetch_rating ASC, a.id DESC\n"
        else:
            base_query += "\nORDER BY au.fetch_rating DESC, a.id DESC\n"
        
        base_query += "OFFSET %s\n"
    
    base_query += "LIMIT %s"

    # execute query
    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
        try:
            if random:
                cur.execute(base_query, (user_id, 
                                        n))
            else:
                cur.execute(base_query, (user_id, offset, n))
        
        except Exception as e:
            print('Could not run fetch articles query, error:', e)
            return []
        else:
            # construct fetchItems object to pass to Jinja2 template
            fetchItems = [{
                "articleId": row["id"],
                "articleUrl": row["url"],
                "title": remove_quotes(row["title"]) if row["title"] else "",
                "date": row["date"], 
                "author_name": row["author_name"], 
                "author_href": row["author_href"],
                "source": row["source"],
                "source_id": row["source_id"],
                "summary": remove_quotes(row["summary"]) if row["summary"] else "",
                "score": row["ai_rating"],
                "user_rating": "false" if row["user_rating"] == None else row["user_rating"],
                "user_read": "true" if row["user_read"] else "false",
            } for row in cur]
    
        if source:
            try:
                cur.execute(source_name_query)
            except Exception as e:
                print('Couldn\'t run source name query, error:', e)
            else:
                source_names = [remove_quotes(row['name']) for row in cur]

    if source:
        return fetchItems, source_names
    else:
        return fetchItems, []

# web URL for login
# message: (optional) message to be passed to login page as to context for why 
@web_app.get("/login", response_class=HTMLResponse)
def loginPage(request: Request, message: str | None = None):
    templates = Jinja2Templates(directory=root_path)
    return templates.TemplateResponse(
        name="login.html", 
        context={"request": request, "message": message}
    )

# endpoint for authentication, passes FORM email & password; if authenticated, returns access token
# redirects either back to login page (if failure) or main page (if success)
# email: email address from login form
# password: (plaintext) password from login form
@web_app.post("/auth")
def authPage(email: str = Form(), password: str = Form()):
    import datetime
    from jose import jwt
    import os

    user = get_user(email)
    if user and pwd_context.verify(password, user.hashed_password):
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30) # 30 day expiry
        data = {
            'email': user.email, 
            'exp': expire
        }
        token = jwt.encode(data, os.environ['SECRET_KEY'], algorithm="HS256")
        response = RedirectResponse("/", status_code=fastapi.status.HTTP_302_FOUND)
        response.set_cookie(key='access_token', value=token, expires=expire)
        return response
    else:
        return RedirectResponse("/login?message=Incorrect+email+or+password", status_code=fastapi.status.HTTP_302_FOUND)

# web URL for main feed, performs initial fetch and serves Preact app via Jinja2 template
# uses FastAPI's Depends injection to require access token in cookie
# note may return an empty feed if there is an issue, template should handle
# n: (default = 10) number of articles to pull (capped at 50)
# unread: (default = True) whether or not to only return unread content
# bad: (default = False) whether or not to return feed in reverse order
# random: (default = False) whether or not to return random feed items in random order
@web_app.get("/")
def mainPage(request: Request, user: User | None = Depends(get_current_user_cookie), n: int=Query(10, le=50), unread: bool=True, bad: bool=False, random: bool=False, source: List[int] = Query(None)):
    if not user:
        return RedirectResponse("/login?message=Please+login")

    fetchItems, source_names = fetchCall(user_id=user.id, n=n, unread=unread, bad=bad, random=random, source=source)
    
    # return template
    templates = Jinja2Templates(directory=root_path)
    return templates.TemplateResponse(
        name="feedview.html.j2", 
        context={
            "userId": user.id, 
            "fetchItems": fetchItems, 
            "n": n, 
            "source": source if source else [],
            "source_names": source_names,
            "bad": "true" if bad else "false", # javascript bools are lowercase
            "random": "true" if random else "false", # javascript bools are lowercase
            "request": request
        }
    )

# GET endpoint for feed
# uses FastAPI's Depends injection to require access token in bearer token
# note may return an empty set of items if there is an issue, requester should handle
# n: (default = 5) number of articles to pull (capped at 50)
# offset: (default = 0) number of articles to skip (to be used for pagination)
# unread: (default = True) whether or not to only return unread content
# bad: (default = False) whether or not to return feed in reverse order
# random: (default = False) whether or not to return random feed items in random order
@web_app.get("/articles", status_code=fastapi.status.HTTP_202_ACCEPTED)
def fetchArticles(n: int = Query(5, le=50), offset: int = 0, unread: bool = True, bad: bool = False, random: bool=False, source: List[int] = Query(None), current_user: User | None = Depends(get_current_user_bearer)) -> FeedItemListResponse:
    if not current_user:
        raise HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    fetchItems, source_names = fetchCall(user_id=current_user.id, n=n, offset=offset, unread=unread, bad=bad, random=random, source=source)
    return {'items': fetchItems, 'offset': offset + n}

# POST endpoint to mark an item as read and updates the user's read history
# uses FastAPI's Depends injection to require access token in bearer token
# payload: POST body that requires an articleId and status (True for read, False for unread)
@web_app.post("/read", status_code=fastapi.status.HTTP_202_ACCEPTED)
def markItemRead(payload: MarkItemReadPayload, current_user: User | None = Depends(get_current_user_bearer)):
    import psycopg2
    from pgvector.psycopg2 import register_vector
    import json

    # pull fetch configuration from JSON file which dictates weights / parameters for
    # recency_factor: (0.25) how much recency matters (should sum with rating_factor and similarity_factor to 1.0)
    # day_decay_threshold: (-60) at what point does article age no longer decay (can be anything from -1 to -inf)
    # day_decay_scale: (30.0) how exponential decay scales (higher number = slower, lower number = faster)
    # rating_factor: (0.5) how much AI rating matters (should sum with recency_factor and similarity_factor to 1.0)
    # similarity_factor: (0.25) how much similarity to user history matters (should sum with recency_factor and rating_factor to 1.0)
    # article_history_inertia: (0.85) inertia factor on recently read history (should be 0.0-1.0)
    # read_articles_to_refresh: (10) how often to refresh cached similarity scores
    with open(root_path + 'fetchparams.json', 'r') as f:
        fetchparams = json.load(f)

    if not current_user:
        raise HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        register_vector(con) # allow python adapter to handle numpy arrays
        cur = con.cursor()
        # update read state
        try:
            cur.execute("UPDATE articleuser SET user_read = %s, updated_at = NOW(), read_timestamp = NOW() WHERE user_id = %s AND article_id = %s", (payload.status, current_user.id, payload.articleId))
            con.commit()
        except Exception as e:
            print('Could not update read status on article', payload.articleId, 'for user', current_user.id, 'to', payload.status, 'due to:', e)
            cur.execute("ROLLBACK")
            raise HTTPException(status_code=424, detail="Could not update read status on article " + str(payload.articleId) + " for user " + str(current_user.id) + " to " + str(payload.status))
        
        # update user history embedding and check refresh
        read_articles_to_refresh = fetchparams["read_articles_to_refresh"]
        old_weight = fetchparams['article_history_inertia']
        new_weight = 1 - old_weight
        try:            
            cur.execute("SELECT a.embedding AS article_embedding, u.recent_articles_read AS user_embedding, u.articles_since_lastrefresh FROM articleuser au JOIN articles a ON au.article_id = a.id JOIN users u ON au.user_id = u.id WHERE au.article_id = %s AND au.user_id = %s",
                (payload.articleId, current_user.id)
            )
            article_embedding, user_read_embedding, user_articles_since_lastrefresh = cur.fetchone()
            if payload.status:
                new_user_read_embedding = user_read_embedding * old_weight + article_embedding * new_weight
            else:
                new_user_read_embedding = (user_read_embedding - article_embedding * new_weight)*(new_weight+old_weight)/old_weight
            
            cur.execute("UPDATE users SET recent_articles_read = %s, updated_at = NOW(), articles_since_lastrefresh = %s WHERE id = %s", (new_user_read_embedding, (user_articles_since_lastrefresh + 1) % read_articles_to_refresh, current_user.id))
            con.commit()

            # if need to trigger refresh, do so
            if (user_articles_since_lastrefresh + 1) % read_articles_to_refresh == 0:
                cur.execute("""UPDATE articleuser
SET article_user_similarity = 0.5*(a.embedding <=> u.recent_articles_read)
FROM articles a, users u
WHERE articleuser.article_id = a.id
AND articleuser.user_id = u.id
AND articleuser.user_id = %s;""", (current_user.id,))
                
                cur.execute("""UPDATE articleuser
SET fetch_rating = CASE 
    WHEN su.always_show = TRUE THEN 100.0 
    ELSE (
        %s * EXP(GREATEST((a.date - CURRENT_DATE)::INT, %s) / %s) + 
        %s * COALESCE(articleuser.user_rating, articleuser.ai_rating) - 
        %s * articleuser.article_user_similarity
    ) 
END
FROM articles a, sourceuser su
WHERE articleuser.article_id = a.id
AND su.user_id = articleuser.user_id
AND su.source_id = a.source
AND articleuser.user_id = %s;""", (fetchparams['recency_factor'], 
                                    fetchparams['day_decay_threshold'],
                                    fetchparams['day_decay_scale'],
                                    fetchparams['rating_factor'],
                                    fetchparams['similarity_factor'],
                                    current_user.id,
                                ))

                con.commit() 

        except Exception as e:
            print('Could not update user', current_user.id, 'recent_articles_read for article', payload.articlId, 'due to:', e)
            cur.execute("ROLLBACK")

# POST endpoint to rate an item
# uses FastAPI's Depends injection to require access token in bearer token
# payload: POST body that requires an articleId and rating (float or None)
@web_app.post("/rate", status_code=fastapi.status.HTTP_202_ACCEPTED)
def rateItem(payload: RateItemPayload, current_user: User | None = Depends(get_current_user_bearer)):
    import psycopg2

    if not current_user:
        raise HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with psycopg2.connect(os.environ['PSYCOPG2_CONNECTION_STRING']) as con:
        cur = con.cursor()
        try:
            cur.execute("UPDATE articleuser SET user_rating = %s, updated_at = NOW() WHERE user_id = %s AND article_id = %s", (payload.rating, current_user.id, payload.articleId))
            con.commit()
        except Exception as e:
            print('Could not update rating on article', payload.articleId, 'for user', current_user.id, 'to', payload.rating, 'due to:', e)
            cur.execute("ROLLBACK")
            raise HTTPException(status_code=424, detail="Could not update rating on article " + str(payload.articleId) + " for user " + str(current_user.id) + " to " + str(payload.rating))

# link FastAPI to Modal
@app.function(volumes={"/data": vol}, secrets=[modal.Secret.from_name("newsreader_psycopg2"), modal.Secret.from_name("newsreader_supabase"), modal.Secret.from_name("newsreader_crypto")])
@modal.asgi_app()
def web():
    web_app.mount(root_path, StaticFiles(directory=root_path))
    return web_app