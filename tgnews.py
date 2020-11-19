# -*- coding: utf-8 -*-

"""News Clustering Baseline."""

import argparse
import itertools
import json
import logging
import multiprocessing
import os
import pickle
import re
import socket
import string
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from enum import Enum, auto
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

import fasttext
import jsonlines
import numpy as np
import scipy
import sentencepiece as spm
import sklearn
from sklearn.cluster import DBSCAN


# Min threshold for language classification (RU).
MIN_LANG_CLF_THRESHOLD_RU = 0.40

# Min threshold for language classification (EN).
MIN_LANG_CLF_THRESHOLD_EN = 0.40

# High confidence threshold in language classification.
HIGH_CONFIDENCE_LANG_THRESHOLD = 0.8

# Language constants.
LANG_EN = 'en'
LANG_RU = 'ru'
LANG_TG = 'tg'
LANG_UA = 'ua'
LANG_BG = 'bg'

# Supported languages.
LANGUAGES = {LANG_EN, LANG_RU}

# Tajik symbols.
TAJIK_SYMBOLS = {'ғ', 'қ', 'ў', 'ӯ', 'ҳ', 'ҷ', 'ї'}

# Min threshold for news classification (using "Other" by default).
MIN_CLF_THRESHOLD = 0.25

# Min threshold for ignoring blacklisted tokens title presence.
BLACKLIST_OVERRIDE_THRESHOLD = 0.8

# Cosine similarity threshold for assigning an article to a cluster.
CLUSTER_ASSIGNMENT_THRESHOLD = 0.4

# Max number of characters to read for an article.
MAX_CHARS = 2500

# Max number of characters to use for a title.
MAX_CHARS_TITLE = 300

# Max number of characters to use for a description.
MAX_CHARS_DESC = 500

# Max size of a news cluster about the same event.
MAX_CLUSTER_SIZE = 750

# Boosting coefficient for tokens which have a heavier clustering weight (e.g. geo).
TOKEN_BOOST_COEFFICIENT = 1.8

# Minimum number of words in a title.
MIN_TITLE_WORD_CNT = 4

# Minimum number of known tokens in a title.
MIN_TITLE_TOKEN_CNT = 5

# Minimum number of characters in a word.
MIN_WORD_LEN = 2

# Max number of files to use.
MAX_FILE_CNT = 10000000

# Max number of files to use.
MAX_INDEX_SIZE = 100000000

# Number of processes to use for parallel file processing.
NUM_PROCESSES = 16

# Max number of article threads to return in a server mode.
MAX_THREADS_CNT = 1000

# Number of HTTP server threads.
HTTP_SERVER_THREAD_CNT = 100

# Classification categories.
ID_TO_CATEGORY = {0:'society', 1:'economy', 2:'technology', 3:'sports', 4:'entertainment', 5:'science', 6:'other'}
CATEGORIES = {'society', 'economy', 'technology', 'sports', 'entertainment', 'science', 'other'}

# Regular expression for the title extraction.
TITLE_REGEX = re.compile('title.\s+content=.([^>]+)>')

# Regular expression for the description extraction.
DESC_REGEX = re.compile('description.\s+content=.([^>]+)>')

# Regular expression for the publication time extraction.
PUBLICATION_TIME_REGEX = re.compile('published_time.\s+content=.([^>]+)>')

# Regular expression for the URL extraction.
URL_REGEX = re.compile('url.\s+content=.([^>]+)>')

# Regular expression for "other" category.
OTHER_REGEX = re.compile('(прогноз погоды|гороскоп|\sпогода|\sпогоду|тельцам|телец|\sовен|\sовнам|девам|стрелец|стрельцам|козерог|козерогу|козерогам|водолей|водолею|водолям|weather|horoscope|met eireann|lotto)\s')

# Regular expression for filtering out discounts.
# NOTE: not used, as apparently ads are allowed?
DISCOUNT_REGEX = re.compile('(\%|\d|percent)\soff\s')

# Regular expression for filtering out savings articles.
# NOTE: not used, as apparently ads are allowed?
SAVINGS_REGEX = re.compile('(save\s(£|\$|€|up))|(just\s(US)?(£|\$|€))|(for\s(£|\$|€))\d{2,3}(\.|$|\s)')

# Regular expression for filtering out sale-related articles.
SALE_REGEX = re.compile('(on|for) sale|(anniversary|apple|huge|amazon|friday|monday|christmas|fragrance|%) sale')

# Regular expression for banned phrases.
BAD_PHRASES_REGEX = re.compile('(смотреть онлайн|можно приобрести|стоит всего|со скидкой|лучшие скидки|составлен топ|простой способ|простейший способ|способа|способов|free download|shouldn\'t miss|of the week|рецепт|правила|the week in)')

# Regular expression for filtering out articles with lists.
LIST_REGEX = re.compile('\d+ (акци|банальн|важн|вещ|вопрос|главн|животн|знаменит|качествен|книг|лайфхак|лучш|мобил|необычн|популяр|привыч|прилож|причин|признак|продукт|прост|професс|самы|способ|технолог|худш|урок|шаг|факт|фильм|экзотичес|adorable|big|beaut|best|creative|crunchy|easy|huge|fantastic|innovative|iconic|baking|inspiring|perfect|stunning|stylish|unconventional|unexpected|wacky|wondeful|worst|habit|event|food|gift|question|reason|sign|step|thing|tip|trick|way)')
LIST_REGEX_STR_START = re.compile('^\d+.{0,16} (акци|банальн|важн|вещ|вопрос|главн|животн|знаменит|качествен|книг|лайфхак|лучш|мобил|необычн|популяр|привыч|прилож|причин|признак|продукт|прост|професс|самы|способ|технолог|худш|урок|шаг|факт|фильм|экзотичес|adorable|big|beaut|best|creative|crunchy|easy|huge|fantastic|innovative|iconic|baking|inspiring|perfect|stunning|stylish|unconventional|unexpected|wacky|wondeful|worst|habit|event|food|gift|question|reason|sign|step|thing|tip|trick|way)')
LIST_REGEX_TOP = re.compile('^(the|top|топ)[\s-]\d+')

# Regular expression for filtering out "How to" guides and informational articles.
HOWTO_REGEX = re.compile('^(best|check out|do you|got a|dont|get|have you|have YOU|here is|here are|how|let us|looking for|look for|my |say|should you|shall you|the best|the well|thi|this|these|useful tips|where|what|who|when|why|want|как|какие|какой|какую|почему|когда|куда|можно|стоит|почему|что\s|кто|с кем|кому|откуда|который|сколько|где|виды|лучшие|самы|зачем)\s')
HOWTO_ENDINGS_REGEX = r'как\s[а-я]+(ти|ть|сь|ся|чь)\b'

# Regular expression for filtering out titles based on their beginnings.
BAD_BEGINNINGS_REGEX = re.compile('^(commentary|review|removed|on this day):')

# Regular expression for repeated spaces.
SPACE_REP_REGEX = re.compile(r' +')

# Regular expression for digits.
DIGIT_REGEX = re.compile(r'\d')

# Regular expression for acronym detection.
ACRONYM_REGEX = re.compile(r'\b([A-ZА-Я]{2,4})\b')

# Punctuation replacement.
STR_TRANSLATE = str.maketrans(' ', ' ', string.punctuation.replace('$', ''))

# Stopwords for RU an EN.
STOPWORDS_EN = {'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'}
STOPWORDS_RU = {'а', 'без', 'более', 'больше', 'будет', 'будто', 'бы', 'был', 'была', 'были', 'было', 'быть', 'в', 'вам', 'вас', 'вдруг', 'ведь', 'во', 'вот', 'впрочем', 'все', 'всегда', 'всего', 'всех', 'всю', 'вы', 'где', 'да', 'даже', 'два', 'для', 'до', 'другой', 'его', 'ее', 'ей', 'ему', 'если', 'есть', 'еще', 'ж', 'же', 'за', 'зачем', 'здесь', 'и', 'из', 'или', 'им', 'иногда', 'их', 'к', 'как', 'какая', 'какой', 'когда', 'конечно', 'кто', 'куда', 'ли', 'лучше', 'между', 'меня', 'мне', 'много', 'может', 'можно', 'мой', 'моя', 'мы', 'на', 'над', 'надо', 'наконец', 'нас', 'не', 'него', 'нее', 'ней', 'нельзя', 'нет', 'ни', 'нибудь', 'никогда', 'ним', 'них', 'ничего', 'но', 'ну', 'о', 'об', 'один', 'он', 'она', 'они', 'опять', 'от', 'перед', 'по', 'под', 'после', 'потом', 'потому', 'почти', 'при', 'про', 'раз', 'разве', 'с', 'сам', 'свою', 'себе', 'себя', 'сейчас', 'со', 'совсем', 'так', 'такой', 'там', 'тебя', 'тем', 'теперь', 'то', 'тогда', 'того', 'тоже', 'только', 'том', 'тот', 'три', 'тут', 'ты', 'у', 'уж', 'уже', 'хорошо', 'хоть', 'чего', 'чем', 'через', 'что', 'чтоб', 'чтобы', 'чуть', 'эти', 'этого', 'этой', 'этом', 'этот', 'эту', 'я'}
STOPWORDS_ALL = STOPWORDS_EN | STOPWORDS_RU

# Title patterns for news filtering.
TITLE_PATTERNS_EN = {'W: W W', 'W W: W W', 'W W W: W', 'W W. W', 'W, W W', 'W\'W W', 'W’W W', 'W W W, W #, ####', 'W - ##W W ####', 'W - #W W ####', 'W W W, W ##, ####', 'W # W #: W', 'W W W W, W. ##', '## W W W W W W: W ##, ####', 'W W W W. ##', '# W W W W W, W ##, ####', 'W, W', 'W W – “W W W”', 'W W (W W)', 'W ## W ####', 'W #,### W', '#####W####W#### W', 'W W W - #W W ####', 'W W W - ##W W ####', 'W W - ##W W ####', 'W W - #W W ####', 'W W W W, W #', 'W W – “W”', 'W | W W-W', 'W\'W W W: W, W ##', 'W (W), W W'}
TITLE_PATTERNS_RU = {'W W: W W', 'W W W?', '№####', 'W W W!', '***', 'W W — W', 'W – W. ##.##.####. W W W W W', 'W!', 'W W!', 'W W W W!', '«W W W W»', '«W W W»', '«W W»', 'W W. W W', 'W–W', 'W-W', 'W W-W', 'W W - W W', 'W W – W W', 'W W...', 'W W W...', 'W W W W...', 'W W? W W W', 'W: W W ## W (W W ##.##.####)', 'W ## W. W W ##.##.####', 'W W W W. ## W', 'W – W. W W. W W', 'W W W ####', 'W W ##.##', 'W W (## W)', 'W: W W - ##', 'W W W? W W W W', 'W W: W W # W (W W ##.##.####)', 'W W: W W W (W W ##.##.####)'}
TITLE_PATTERNS_OTHER_EN = {'W W W W, W #, ####', 'W W, W W', 'W W W W \'W # W\' W', 'W W W W – W ##', 'W W W W W, W #', 'W W W W ## – ##', 'W\'W W W W: W', 'W W W - W, W ##, ####', 'W W W - W, W #, ####', 'W W W W- W #', 'W W W W #:## W.W. W', 'W #: W W W'}
TITLE_PATTERNS_OTHER_RU = {'W W W ##.##.##', 'W W W W: W, ## W', 'W W W W: W, # W', 'W W ##.##.####', 'W W. W W ##.##.####', 'W W W W, ## W', 'W W ##.##.##', 'W W ## W #### W W W W', 'W W W ## W #### W.', '«W. W W» #.##.####', '«W. W W» ##.##.####', '«W. W W» W ##. ##.##.####', '«W. W W» W ##. #.##.####', 'W W. W W ##.##.#### (##.##)', 'W W (W W ##.##.####)', 'W W W W # W (W)', 'W W W W W, ## W #### W', 'W W W ## W, W', 'W. W W ##.##.####', 'W W W W W ## W ####', 'W W ## W #### W', 'W W W W – ## W ####', 'W W ## W ####: W W W', 'W W W (## W)', 'W W W W W, ## W', 'W W W, ## W ####', 'W W W W W, # W', 'W W # W ####', '## W: W W, W W W W', 'W W W. W W ## W', '## W. W W', '«W. W» ## W ####', 'W W. ## W', 'W W: # W ####', 'W W W W # W ####', 'W W # W #### W W W W'}

# News categories for tokens in the URL.
TOKEN_CATEGORIES = {'accidents':'society','crime':'society','geopolitics':'society','incident':'society','incidents':'society','politics':'society','politika':'society','business':'economy','economy':'economy','economic':'economy','economics':'economy','ekonomika':'economy','finance':'economy','markets':'economy','money':'economy','stocks':'economy','baseball':'sports','basketball':'sports','cricket':'sports','football':'sports','football-news':'sports','futbol':'sports','rugby':'sports','soccer':'sports','sport':'sports','sports':'sports','tennis':'sports','bollywood':'entertainment','entertainment':'entertainment','movies':'entertainment','showbiz':'entertainment','health':'science','science':'science','technology':'technology'}
TOKEN_CATEGORIES_SET = set(TOKEN_CATEGORIES.keys())

# Min time for time bucket calculation (datetime(2019, 11, 1)).
MIN_BUCKET_DATE = 1572570000

# Possible server states.
class ServerState(Enum):
    OFF = auto()
    LOADING = auto()
    READY = auto()

# Permitted CLI commands.
CLI_MODE_COMMANDS = {'languages', 'news', 'categories', 'threads', 'top'}
CLI_COMMANDS = CLI_MODE_COMMANDS | {'server'}

# Names of fields in a server response for a GET request for threads.
THREAD_FIELDS = ['title', 'category', 'articles']

# List of fields to check in order to determine whether article has changed.
ARTICLE_MUTABLE_FIELDS = {'file_name', 'original_title', 'desc', 'language', 'publication_time', 'domain', 'ttl'}

# Current state of the HTTP server.
CURRENT_SERVER_STATE = ServerState.OFF

# TTL constants.
TTL_NO_EXPIRATION = 100 * 365 * 24 * 60 * 60
TTL_MAX = 30 * 24 * 60 * 60
TTL_MIN = 5 * 60
TTL_CLEANUP_FREQUENCY = 60

# Set of filenames of all articles (including non-en/ru and non-news).
all_articles = set()

# Mapping from article file name to article info.
file_name_to_article = {}

# Publication time of latest article (datetime(2000, 1, 1)).
max_publication_time = 946688400

# Mapping from (time bucket ID, lang) to its info:
# {'start_time', 'end_time', 'article_ids', 'mx', 'clusters', 'reverse_ix'}.
time_buckets = {}

# Times of last TTL cleanup and index disk dump.
last_ttl_cleanup_time = time.time()

# Lock for multi-threading processing of server requests.
lock = threading.RLock()

# Logging setup.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

LOGGING_FORMAT = '%(asctime)-15s: %(message)s'
formatter = logging.Formatter(LOGGING_FORMAT)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('tgnews.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Index I/O.
def create_json_writer(file_path, mode='a', compact=True, sort_keys=True, flush=True):
    """Create a `jsonlines` writer."""
    return jsonlines.Writer(open(file_path, mode), compact=compact, sort_keys=sort_keys, flush=flush)

ARTICLE_UPDATES_PATH = 'article_updates.jsonl'
ARTICLE_UPDATES_FILE = create_json_writer(ARTICLE_UPDATES_PATH)

# Article info fields which are saved to disk.
SAVED_ARTICLE_FIELDS = {'file_name', 'publication_time', 'time_bucket_id', 'ttl', 'language', 'domain', 'domain_pr', 'category', 'original_title', 'index_row_id', 'cluster_id'}


# Article update event types.
class ArticleUpdate(Enum):
    ADD = auto()
    DELETE = auto()
    IGNORE = auto()


def lowercase_except_acronyms(input_str):
    """Convert a string to a lower case (except acronyms)."""
    result_str = input_str.lower()
    
    # No way of knowing if anything is an acronym.
    if input_str.isupper():
        return result_str
    
    for m in ACRONYM_REGEX.finditer(input_str):
        start = m.start()
        end = m.start() + len(m.group())
        result_str = result_str[:start] + result_str[start:end].upper() + result_str[end:]
    return result_str


def clean_string(input_str):
    """Prepare an input string for TF-IDF and classification."""
    input_str = lowercase_except_acronyms(input_str.strip()).replace('&quot;', '').replace(
        '\xa0', ' ').replace('\xad', '').replace('«', '').replace('»', '').replace(
        '‘', '').replace('’', ' ').replace('-', ' ').replace('–', ' ').replace('£', ' £ ').replace(
        '₹', ' ₹ ').replace('€', ' € ').replace('$', ' $ ').replace(',', '').replace(
        '.', '').translate(STR_TRANSLATE)

    input_str = re.sub(SPACE_REP_REGEX, ' ', input_str)
    input_str = ' '.join([word2norm.get(x, x) for x in input_str.split(' ')])
    return input_str


def get_hour_bucket_id(input_datetime, start_time=MIN_BUCKET_DATE, bucket_size_sec=57600):
    """Get hour bucket ID since the beginning of time."""
    input_datetime = max(start_time, input_datetime)
    return int((input_datetime - start_time) // bucket_size_sec)


def replace_digits(input_str):
    """Replace digits for title pattern generation."""
    return '#' * len(input_str.group())


def get_title_pattern(input_str):
    """Get a title punctuation pattern."""
    return re.sub(r'\w+', 'W', re.sub(r'\d+', replace_digits, input_str))


def process_html(html_text, file_name):
    """Extract news article information from HTML content."""
    title_search_res = TITLE_REGEX.search(html_text)

    title = title_search_res.group(1)[:-2].strip() if title_search_res else ''
    original_title = title
    original_title = original_title.replace('&amp;', '&').replace('&quot;', '"').replace('&apos;', "'")
    title_with_digits = clean_string(title[:MAX_CHARS_TITLE])
    title = re.sub(DIGIT_REGEX, '', title_with_digits)

    url_search_res = URL_REGEX.search(html_text)
    url = url_search_res.group(1)[:-2].strip() if url_search_res else ''
    domain = urlparse(url).netloc if url else ''

    desc_search_res = DESC_REGEX.search(html_text)
    desc = desc_search_res.group(1)[:-2].strip() if desc_search_res else ''
    desc_raw = original_title + ' ' + desc
    desc = clean_string(desc[:MAX_CHARS_DESC])
    desc_lang = (title + ' ' + desc)
    desc = (title + ' ' + domain + ' ' + desc)

    lang_res = lang_model.predict(desc_lang)
    language = lang_res[0][0][9:]
    language_score = lang_res[1][0]

    if ((language == LANG_RU and language_score < MIN_LANG_CLF_THRESHOLD_RU)
        or (language == LANG_EN and language_score < MIN_LANG_CLF_THRESHOLD_EN)):
        language = 'unk'

    # Tajik fix (often leads to FPs in RU).
    if language == LANG_RU:
        if (language_score < HIGH_CONFIDENCE_LANG_THRESHOLD
            and (TAJIK_SYMBOLS & set(desc_raw.lower()))):
            language = LANG_TG
        elif '(укр)' in desc_raw or '(укр.)' in desc_raw:
            language = LANG_UA
        elif '.bg' in desc_raw:
            language = LANG_BG

    publish_time_search_res = PUBLICATION_TIME_REGEX.search(html_text)
    publication_time = int(datetime.strptime(publish_time_search_res.group(1)[:-8], '%Y-%m-%dT%H:%M:%S').timestamp()) \
                       if publish_time_search_res else MIN_BUCKET_DATE

    return {
        'file_name': file_name,
        'title': title,
        'title_with_digits': title_with_digits,
        'title_pattern': get_title_pattern(original_title),
        'desc': desc,
        'url': url,
        'domain': domain,
        'domain_pr': domain_pagerank.get(domain, 0),
        'original_title': original_title,
        'language': language,
        'publication_time': publication_time,
        'time_bucket_id': get_hour_bucket_id(publication_time),
        'ttl': TTL_NO_EXPIRATION
    }


def process_file(file_path):
    """Extract news article information from a file."""
    try:
        with open(file_path, 'r') as f:
            html_text = f.read(MAX_CHARS)
        file_name = os.path.basename(file_path)
        return process_html(html_text, file_name)
    except Exception as e:
        pass


def get_article_info(source_dir):
    """Fetch article info for a given directory."""
    file_paths = []
    for root, subdirs, files in os.walk(source_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.endswith('.html'):
                file_paths.append(file_path)

    p = multiprocessing.Pool(NUM_PROCESSES)
    articles = p.map(process_file, file_paths[:MAX_FILE_CNT])
    p.close()

    return [x for x in articles if x]


def get_languages_response(article_info):
    """Prepare a response for the `languages` command."""
    lang_articles = defaultdict(list)
    for a in article_info:
        lang_articles[a['language']].append(a['file_name'])

    response = []
    for lang_code, articles in lang_articles.items():
        if lang_code in LANGUAGES:
            response.append({'lang_code': lang_code, 'articles': articles})
    for lang_code, articles in lang_articles.items():
        if lang_code not in LANGUAGES:
            response.append({'lang_code': lang_code, 'articles': articles})
    return response


def classify_news(article_info):
    """Classify news into one of the 7 categories."""
    def process_articles(articles, tfidf, clf):
        if not articles:
            return

        # Vectorize title + description and get a number of words found in vocab.
        desc_vec = tfidf.transform([a['desc'] for a in articles])
        
        # Get classifer predictions.
        pred_proba = clf.predict_proba(desc_vec)
        pred = np.argmax(pred_proba, axis=1)

        # Get URL token predictions.
        url_preds = []
        for a in articles:
            candidate_categories = {TOKEN_CATEGORIES[t] for t in
                set(a['url'].split('/')) & TOKEN_CATEGORIES_SET}
            if len(candidate_categories) == 1:
                url_preds.append(list(candidate_categories)[0])
            else:
                url_preds.append('')

        # Combine URL tokens and classifier predictions.
        categories = []
        for pp, p, url_p, a in zip(pred_proba, pred, url_preds, articles):
            if ((a['language'] == LANG_EN and a['title_pattern'] in TITLE_PATTERNS_OTHER_EN)
                or (a['language'] == LANG_RU and a['title_pattern'] in TITLE_PATTERNS_OTHER_RU)):
                categories.append('other')
            elif re.search(OTHER_REGEX, a['original_title'].lower()):
                categories.append('other')
            else:
                if url_p and pp[p] < 0.8:
                    pp[p] = 1.01  # Used later for news filtering.
                    categories.append(url_p)
                else:
                    categories.append(ID_TO_CATEGORY[6 if pp[p] < MIN_CLF_THRESHOLD else p])

        desc_vocab_token_cnts = desc_vec.getnnz(1)  # Number of known tokens.
        for a, c, pp, wc in zip(articles, categories, pred_proba, desc_vocab_token_cnts):
            a['category'] = c
            a['category_confidence'] = max(pp)
            a['vocab_word_cnt'] = wc

    articles_en = [a for a in article_info if a['language'] == LANG_EN]
    articles_ru = [a for a in article_info if a['language'] == LANG_RU]

    process_articles(articles_en, vectorizer_en, clf_en)
    process_articles(articles_ru, vectorizer_ru, clf_ru)

    return article_info


def filter_news(article_info):
    """Filter out non-news articles."""
    news_articles = []
    for a in article_info:
        title_lower = a['original_title'].lower()
        title_words = a['title_with_digits'].split(' ')
        title_words_set = set(x for x in title_words if len(x) >= MIN_WORD_LEN or re.search(r'в|о|с|у|\d', x))

        # Allow articles which have whitelisted tokens or get confidently classified.
        is_whitelisted = (len(tokens_all & title_words_set) > 0
                          or (a['category_confidence'] >= MIN_CLF_THRESHOLD))

        # Ignore articles which contain blacklisted tokens (ignore high-confidence cases).
        is_not_blacklisted = ((len(tokens_blacklist_all & title_words_set) == 0)
                              or a['category_confidence'] > BLACKLIST_OVERRIDE_THRESHOLD)

        # # Ignore discount and savings titles.
        # is_not_discount = not (re.search(DISCOUNT_REGEX, a['title']) or re.search(SAVINGS_REGEX, title_lower))

        # # Ignore sale titles.
        # is_not_sale = not re.search(SALE_REGEX, a['title'])

        # Ignore "How to" articles.
        is_not_howto = not re.search(HOWTO_REGEX, a['title_with_digits'])
        is_not_howto = is_not_howto and not (re.search(HOWTO_ENDINGS_REGEX, a['original_title']) and len(tokens_all & title_words_set) == 0)

        # Ignore articles with bad phrases in titles.
        is_not_bad_phrase = not re.search(BAD_PHRASES_REGEX, a['title'])

        # Ignore articles based on how their titles begin.
        is_not_bad_beginnings = not re.search(BAD_BEGINNINGS_REGEX, a['original_title'])

        # Ignore articles with lists (such as "10 tips for...").
        is_not_list = not re.search(LIST_REGEX, title_lower)
        is_not_list_str_start = not re.search(LIST_REGEX_STR_START, title_lower)
        is_not_list_top = not re.search(LIST_REGEX_TOP, title_lower)

        # Ignore abnormally short titles.
        is_not_low_word_cnt = len(title_words_set) >= MIN_TITLE_WORD_CNT

        # Ignore mid-short titles with no whitelisted terms.
        is_not_mid_low_word_cnt = not (len(title_words_set) == 4 and len(tokens_all & title_words_set) == 0)
            
        # Ignore titles with low vocab token count.
        is_not_low_vocab_word_cnt = a['vocab_word_cnt'] >= MIN_TITLE_TOKEN_CNT

        # Ignore low-confidence questions.
        is_not_low_quality_question = not (title_lower and (title_lower[-1] == '?') and (a['category_confidence'] < 0.75))

        # Ignore "Artist - ItemName" titles.
        is_not_dash = not ('—' in a['original_title'] and len(tokens_all & title_words_set) == 0 and len(title_words) <= 5)

        # Ignore titles with a slash.
        is_not_slash = not (' / ' in a['original_title'])

        # Ignore exclamations.
        is_not_exclamation = not (a['original_title'][-1] == '!' and len(title_words_set) < 8)

        # Ignore ellipses.
        is_not_ellipsis = not (a['original_title'][-3:] == '...' and a['category_confidence'] < 0.8 and len(title_words_set) < 7)

        # Ignore full quotes.
        is_not_quote = not (a['original_title'][0] == '«' and a['original_title'][-1] == '»' and a['original_title'].count('«') == 1 and a['original_title'].count('»') == 1)

        # Ignore patterns which follow a certain template.
        is_not_templated_pattern = not (
            (a['language'] == LANG_EN and a['title_pattern'] in TITLE_PATTERNS_EN)
            or (a['language'] == LANG_RU and a['title_pattern'] in TITLE_PATTERNS_RU))

        if (is_whitelisted and is_not_blacklisted and is_not_howto and is_not_bad_phrase
            and is_not_bad_beginnings and is_not_list and is_not_list_str_start and is_not_list_top
            and is_not_low_word_cnt and is_not_mid_low_word_cnt and is_not_low_vocab_word_cnt
            and is_not_low_quality_question and is_not_dash and is_not_slash
            and is_not_exclamation and is_not_ellipsis and is_not_quote and is_not_templated_pattern):
            news_articles.append(a)

    return news_articles


def get_filter_news_response(article_info):
    """Prepare a response for the `news` command."""
    return {'articles': [a['file_name'] for a in article_info]}


def get_classify_news_response(article_info):
    """Prepare a response for the `categories` command."""
    result = {x: [] for x in ID_TO_CATEGORY.values()}
    for a in article_info:
        result[a['category']].append(a['file_name'])

    return [{'category': c, 'articles': articles}
            for c, articles in result.items()]


def create_time_bucket(bucket_id):
    """Create a time bucket with pre-filled values."""
    return {
        'id': bucket_id,
        'max_time': MIN_BUCKET_DATE,
        'mx': np.empty(0),
        'clusters': {},  # Map from article ID to its cluster.
        'inverted_ix': scipy.sparse.lil_matrix(
            (len(vectorizer_en.vocabulary_), MAX_INDEX_SIZE), dtype=np.int32)
    }


def get_time_bucket_info(time_bucket_id, lang_code):
    """Get info about articles and clusters in a given time bucket."""
    tb_key = (time_bucket_id, lang_code)
    if tb_key not in time_buckets:
        time_buckets[tb_key] = create_time_bucket(time_bucket_id)
    return time_buckets[tb_key]


def index_article(file_name, article_info, vectorizer):
    """Add article to the index."""
    global last_ttl_cleanup_time, max_publication_time

    article_lang = article_info['language']
    file_name_to_article[file_name] = article_info

    # Update max publication time.
    max_publication_time = max(article_info['publication_time'], max_publication_time)

    # Get info on current and adjacent time buckets.
    tb_back = get_time_bucket_info(article_info['time_bucket_id'] - 1, article_lang)
    tb_now = get_time_bucket_info(article_info['time_bucket_id'], article_lang)
    tb_forward = get_time_bucket_info(article_info['time_bucket_id'] + 1, article_lang)

    # Transform article title.
    title = article_info['title_with_digits']
    title_tfidf = vectorizer.transform([title])
    term_ix = [int(x) for x in title_tfidf.indices if vectorizer.idf_[x] > 3.5]

    # Boost importance of certain tokens (such as geo).
    matched_tokens = set(title.split(' ')) & boosted_tokens
    for tok in matched_tokens:
        token_ids = (sp_en if article_lang == LANG_EN else sp_ru).encode_as_ids(tok)
        for token_id in token_ids:
            title_tfidf[0, vectorizer.vocabulary_[token_id]] *= TOKEN_BOOST_COEFFICIENT
    title_tfidf = sklearn.preprocessing.normalize(title_tfidf)

    # Get inverted index candidates from current and adjacent time buckets.
    bucket_set = [tb_back, tb_now, tb_forward]
    bucket_set_candidates = []
    candidate_mxs = []
    for idx, bucket in enumerate(bucket_set):
        inverted_index_matches = list(set(itertools.chain.from_iterable(
            bucket['inverted_ix'][term_ix].data)))
        if inverted_index_matches:
            bucket_set_candidates += [(idx, x) for x in inverted_index_matches]
            candidate_mxs.append(bucket['mx'][inverted_index_matches, :])

    # Get article with max similarity to the current article.
    max_sim = 0
    bucket_id = 1
    if bucket_set_candidates:
        candidate_mx = scipy.sparse.vstack(candidate_mxs, format='csr')
        emb_sim = candidate_mx * title_tfidf.T
        local_max_sim_row_id = np.argmax(emb_sim)
        max_sim = max(emb_sim[local_max_sim_row_id])
        if max_sim > CLUSTER_ASSIGNMENT_THRESHOLD:
            bucket_id, cluster_id = bucket_set_candidates[local_max_sim_row_id]
    target_bucket = bucket_set[bucket_id]

    # Update time bucket ID in case article gets appended to an adjacent bucket.
    article_info['time_bucket_id'] = target_bucket['id']

    # Save matrix row ID for further lookups.
    row_id = np.shape(target_bucket['mx'])[0]
    article_info['index_row_id'] = row_id

    # Update article embedding index.
    if np.shape(target_bucket['mx'])[0] == 0:
        target_bucket['mx'] = scipy.sparse.csr_matrix(title_tfidf)
    else:
        target_bucket['mx'] = scipy.sparse.vstack(
            [target_bucket['mx'], title_tfidf], format='csr')

    # Update inverted index.
    target_bucket['inverted_ix'][term_ix, row_id] = row_id

    # Update time bucket max time.
    target_bucket['max_time'] = max(
        article_info['publication_time'],
        target_bucket['max_time'])

    # Attach article to the existing cluster.
    if max_sim > CLUSTER_ASSIGNMENT_THRESHOLD:
        current_cluster = target_bucket['clusters'][cluster_id]
        
        article_ids = current_cluster['article_ids']
        article_file_names = current_cluster['article_file_names']
        article_times = current_cluster['article_times']
        article_categories = current_cluster['article_categories']

        article_ids += (row_id,)
        article_file_names += (file_name,)
        article_times += (article_info['publication_time'],)
        article_categories += (article_info['category'],)
        current_cluster['category'] = Counter(
            current_cluster['article_categories']).most_common(1)[0][0]

        # Sort articles by relevance.
        cluster_mx = target_bucket['mx'][article_ids, :]
        relevance_scores = (cluster_mx * cluster_mx.T).sum(axis=0).A[0]
        _, article_ids, article_file_names, article_times, article_categories = zip(*sorted(zip(
            relevance_scores, article_ids, article_file_names, article_times, article_categories), key=lambda x: -x[0]))

        current_cluster['article_ids'] = article_ids
        current_cluster['article_file_names'] = article_file_names
        current_cluster['article_times'] = article_times
        current_cluster['article_categories'] = article_categories
        current_cluster['max_time'] = max(article_info['publication_time'], current_cluster['max_time'])

        target_bucket['clusters'][row_id] = current_cluster
        article_info['cluster_id'] = current_cluster['id']
    # Create a new cluster with current article in it.
    else:
        current_cluster = {}
        current_cluster['id'] = row_id
        current_cluster['article_ids'] = (row_id,)
        current_cluster['article_file_names'] = (file_name,)
        current_cluster['article_times'] = (article_info['publication_time'],)
        current_cluster['article_categories'] = (article_info['category'],)
        current_cluster['max_time'] = article_info['publication_time']
        current_cluster['category'] = article_info['category']
        target_bucket['clusters'][row_id] = current_cluster
        article_info['cluster_id'] = row_id

    current_time = time.time()
    if current_time - last_ttl_cleanup_time >= TTL_CLEANUP_FREQUENCY:
        run_ttl_cleanup()
        last_ttl_cleanup_time = current_time

    # Save updates to disk.
    ARTICLE_UPDATES_FILE.write({
        'eventType': ArticleUpdate.ADD.name,
        'file_name': file_name,
        'article_info': {k:v for k, v in article_info.items() if k in SAVED_ARTICLE_FIELDS},
        'time_bucket_id': target_bucket['id'],
        'term_ix': term_ix,
        'tfidf_data': title_tfidf.data.tolist(),
        'tfidf_indices': title_tfidf.indices.tolist(),
    })


def get_index_threads(period, lang_code, category):
    """Extract article threads from the index."""
    threads = []
    min_time = max_publication_time - period
    for (bucket_id, bucket_lang_code), info in time_buckets.items():
        if info['max_time'] < min_time:
            continue

        if lang_code != bucket_lang_code:
            continue

        processed_clusters = set()
        for cluster_id, cluster in info['clusters'].items():
            if category != 'any' and cluster['category'] != category:
                continue

            if cluster['max_time'] < min_time:
                continue

            if cluster['id'] in processed_clusters:
                continue

            file_names = cluster['article_file_names']

            title = file_name_to_article[file_names[0]]['original_title']

            unique_pr = {(file_name_to_article[f_name]['domain_pr'],
                          file_name_to_article[f_name]['domain']) for f_name in file_names}
            thread = {
                'title': title,
                'articles': list(file_names),
                'total_pr': sum(x[0] for x in unique_pr),
                'domain_cnt': len(unique_pr)
            }
            if category == 'any':
                thread['category'] = cluster['category']
            threads.append(thread)

            processed_clusters.add(cluster['id'])

    # Sort threads by combined PageRank of articles and article count.
    threads = sorted(threads, key=lambda x: (x['domain_cnt'] * x['total_pr'], len(x['articles'])), reverse=True)

    # Return only `MAX_THREADS_CNT` threads.
    threads = threads[:MAX_THREADS_CNT]

    return {'threads': [{f:t[f] for f in THREAD_FIELDS if f in t} for t in threads]}


def delete_csr_row(mx, i):
    """Delete a row from a CSR matrix."""
    n = mx.indptr[i + 1] - mx.indptr[i]
    if n == 0:
        return
    
    mx.data[mx.indptr[i]:-n] = mx.data[mx.indptr[i+1]:]
    mx.data = mx.data[:-n]

    mx.indices[mx.indptr[i]:-n] = mx.indices[mx.indptr[i+1]:]
    mx.indices = mx.indices[:-n]

    mx.indptr[i+1:] -= n


def delete_article(file_name):
    """Delete an article by file name.

    Note that deletes in the inverted index
    are ignored for performance reasons.
    This has no impact on the clustering output
    as deletes are performed on the main (CSR) matrix.
    """
    global max_publication_time
    all_articles.remove(file_name)

    # Delete EN/RU news article from the index.
    article_info = file_name_to_article.get(file_name, None)
    if article_info:
        if article_info['publication_time'] >= max_publication_time:
            max_publication_time = max([x['publication_time'] for x in file_name_to_article.values() if x['file_name'] != file_name] or [MIN_BUCKET_DATE])
        
        row_id = article_info['index_row_id']
        time_bucket = time_buckets[(article_info['time_bucket_id'], article_info['language'])]

        # Remove a record from CSR matrix with article embedding.
        delete_csr_row(time_bucket['mx'], row_id)

        # Clean up cluster info (for other articles to refer to updated info).
        cluster = time_bucket['clusters'][row_id]
        filter_res = tuple(zip(*filter(lambda x: x[0] != row_id, zip(
            cluster['article_ids'], cluster['article_file_names'], cluster['article_categories'], cluster['article_times']))))
        if filter_res:
            cluster['article_ids'], cluster['article_file_names'], cluster['article_categories'], cluster['article_times'] = filter_res
            cluster['time'] = min(cluster['article_times'])
            cluster['category'] = Counter(
                cluster['article_categories']).most_common(1)[0][0]

        # Remove mapping from article row ID to cluster info.
        time_bucket['clusters'].pop(row_id)

        file_name_to_article.pop(file_name)

    ARTICLE_UPDATES_FILE.write({
        'eventType': ArticleUpdate.DELETE.name,
        'file_name': file_name
    })


def run_ttl_cleanup():
    """Delete articles with expired TTL from an index."""
    articles_to_delete = []
    for file_name, article_info in file_name_to_article.items():
        time_diff = max_publication_time - article_info['publication_time']
        if time_diff > article_info['ttl']:
            articles_to_delete.append(file_name)
    list(map(delete_article, articles_to_delete))


def extract_threads(article_info):
    """Extract news clusters."""
    result = []

    def process_articles(article_info, vectorizer):
        if not article_info:
            return []

        # Get a TF-IDF matrix.
        titles = [a['title_with_digits'] for a in article_info]
        titles_tfidf = vectorizer.transform(titles)

        # Boost importance of certain tokens (such as geo).
        for idx, t in enumerate(titles):
            matched_tokens = set(t.split(' ')) & boosted_tokens
            for tok in matched_tokens:
                token_ids = sp_ru.encode_as_ids(tok)
                for token_id in token_ids:
                    titles_tfidf[idx, vectorizer.vocabulary_[token_id]] *= TOKEN_BOOST_COEFFICIENT

        # Cluster on a local TF-IDF.
        clusterer = DBSCAN(min_samples=2, eps=0.55, metric='cosine')
        clusterer.fit(titles_tfidf)
        clusters = clusterer.labels_.tolist()

        # Cluster assignment.
        cluster_dict = {i: [] for i in range(-1, max(clusters) + 1)}
        for idx, cluster_id in enumerate(clusters):
            cluster_dict[cluster_id].append(idx)

        for cluster_id, article_ids in cluster_dict.items():
            # Ignore "undefined" cluster.
            if cluster_id == -1:
                continue

            # Ignore abnormally large clusters.
            if len(article_ids) > MAX_CLUSTER_SIZE:
                continue

            # Compute cluster-level relevance scores.
            similarity_mx = np.multiply(
                titles_tfidf[article_ids, :],
                np.transpose(titles_tfidf[article_ids, :]))
            relevance_scores = np.asarray(np.sum(similarity_mx, axis=0)).reshape((-1))

            # Compute relative article freshness.
            distances_time = {}
            min_time = min([article_info[a_id]['publication_time'] for a_id in article_ids])
            for a_id in article_ids:
                a_time = article_info[a_id]['publication_time']
                distances_time[a_id] = (a_time - min_time) // 60

            # Filter out articles which are too far from the centroid.
            clean_articles = []
            for idx, a_id in enumerate(article_ids):
                if distances_time[a_id] < (60 * 24):
                    clean_articles.append((
                        a_id,
                        relevance_scores[idx],
                        article_info[a_id]['domain_pr'],
                        distances_time[a_id]))

            # Sort by relevance, PageRank, and freshness.
            # Avoid consecutive articles by the same publisher.
            articles_result = []
            unused_articles = sorted(clean_articles, key=lambda x: (-x[1], -x[2], x[3]))
            while unused_articles:
                prev_domain = 'N/A'
                new_unused_articles = []
                for a in unused_articles:
                    a_domain = article_info[a[0]]['domain']
                    if (not a_domain) or (a_domain != prev_domain):
                        articles_result.append(a)
                    else:
                        new_unused_articles.append(a)
                    prev_domain = a_domain
                unused_articles = new_unused_articles

            # Prepare response, use top article title for naming a thread.
            if len(articles_result) >= 1:
                unique_pr = {(article_info[a[0]]['domain_pr'], article_info[a[0]]['domain']) for a in articles_result}
                result.append({
                    'title': article_info[articles_result[0][0]]['original_title'],
                    'articles': [article_info[a[0]]['file_name'] for a in articles_result],
                    'article_categories': [article_info[a[0]]['category'] for a in articles_result],
                    'domain_cnt': len({article_info[a[0]]['domain'] for a in articles_result}),
                    'total_pr': sum(x[0] for x in unique_pr),
                })

        # Add clusters of size 1 (unassigned articles).
        if -1 in cluster_dict:
            for article_id in cluster_dict[-1]:
                result.append({
                    'title': article_info[article_id]['original_title'],
                    'articles': [article_info[article_id]['file_name']],
                    'article_categories': [article_info[article_id]['category']],
                    'domain_cnt': 1,
                    'total_pr': article_info[article_id]['domain_pr']
                })

    articles_en = [a for a in article_info if a['language'] == LANG_EN]
    articles_ru = [a for a in article_info if a['language'] == LANG_RU]

    process_articles(articles_en, vectorizer_sentpiece_en)
    process_articles(articles_ru, vectorizer_sentpiece_ru)

    return result


def get_threads_response(threads_info, fields=['title', 'articles']):
    """Prepare a response for the `threads` command."""
    return [{f: x[f] for f in fields} for x in threads_info]


def sort_threads(article_threads):
    """Sort threads by relative importance."""
    
    # Determine category of each thread.
    for t in article_threads:
        t['category'] = Counter(t['article_categories']).most_common(1)[0][0]

    # Sort threads by combined PageRank of articles and article count.
    article_threads = sorted(article_threads, key=lambda x: (x['domain_cnt'] * x['total_pr'], len(x['articles'])), reverse=True)

    # Drop irrelevant fields.
    article_threads = get_threads_response(article_threads, ['title', 'articles', 'category'])
    result = [{
        'category': 'any',
        'threads': article_threads
    }]

    # Get threads for each of the categories.
    for _, c in sorted(ID_TO_CATEGORY.items(), key=lambda x: x[0]):
        category_threads = [t for t in article_threads if t['category'] == c]
        category_threads = get_threads_response(category_threads, ['title', 'articles'])
        result.append({
            'category': c,
            'threads': category_threads
        })

    return result


def maybe_load_from_disk():
    """Load index contents from disk."""
    global all_articles, file_name_to_article, max_publication_time, time_buckets

    # Create a set of deleted articles which will be ignored for index construction.    
    deleted_articles = set()
    if os.path.exists(ARTICLE_UPDATES_PATH):
        with jsonlines.open(ARTICLE_UPDATES_PATH, 'r') as f:
            for article_update in f:
                if article_update['eventType'] == ArticleUpdate.DELETE.name:
                    deleted_articles.add(article_update['file_name'])
                elif article_update['eventType'] in {
                ArticleUpdate.ADD.name, ArticleUpdate.IGNORE.name}:
                    if article_update['file_name'] in deleted_articles:
                        deleted_articles.remove(article_update['file_name'])

    # Fetch index-related data.
    max_publication_time_local = max_publication_time
    all_articles_local = set()
    file_name_to_article_local = {}
    time_buckets_local = {}
    clusters_local = {}
    time_bucket_tfidf_info = defaultdict(lambda : {'row':[], 'col':[], 'data': []})
    time_bucket_inverted_ix_info = defaultdict(lambda : {'row':[], 'col':[], 'data': []})
    if os.path.exists(ARTICLE_UPDATES_PATH):
        with jsonlines.open(ARTICLE_UPDATES_PATH, 'r') as f:
            for article_update in f:
                if article_update['file_name'] in deleted_articles:
                    continue

                all_articles_local.add(article_update['file_name'])

                if article_update['eventType'] == ArticleUpdate.ADD.name:
                    article_info = article_update['article_info']

                    max_publication_time_local = max(
                        article_info['publication_time'],
                        max_publication_time_local)

                    file_name_to_article_local[article_update['file_name']] = article_info

                    # Fetch time bucket.
                    tb_key = (article_info['time_bucket_id'], article_info['language'])
                    if tb_key not in time_buckets_local:
                        time_buckets_local[tb_key] = create_time_bucket(article_info['time_bucket_id'])

                    # Update bucket clusters.
                    local_cluster_id = (tb_key, article_info['cluster_id'])
                    if local_cluster_id in clusters_local:
                        current_cluster = clusters_local[local_cluster_id]
                    else:
                        current_cluster = {}
                        current_cluster['id'] = article_info['cluster_id']
                        current_cluster['article_ids'] = tuple()
                        current_cluster['article_file_names'] = tuple()
                        current_cluster['article_times'] = tuple()
                        current_cluster['article_categories'] = tuple()
                        clusters_local[local_cluster_id] = current_cluster

                    current_cluster['article_ids'] += (article_info['index_row_id'],)
                    current_cluster['article_file_names'] += (article_info['file_name'],)
                    current_cluster['article_times'] += (article_info['publication_time'],)
                    current_cluster['article_categories'] += (article_info['category'],)
                    time_buckets_local[tb_key]['clusters'][article_info['index_row_id']] = current_cluster

                    # Update time bucket max time.
                    time_buckets_local[tb_key]['max_time'] = max(
                        article_info['publication_time'],
                        time_buckets_local[tb_key]['max_time'])

                    # Save TF-IDF data.
                    for col, data in zip(
                        article_update['tfidf_indices'],
                        article_update['tfidf_data']):
                        time_bucket_tfidf_info[tb_key]['row'].append(article_info['index_row_id'])
                        time_bucket_tfidf_info[tb_key]['col'].append(col)
                        time_bucket_tfidf_info[tb_key]['data'].append(data)

                    # Save inverted index data.
                    for ix in article_update['term_ix']:
                        time_bucket_inverted_ix_info[tb_key]['row'].append(ix)
                        time_bucket_inverted_ix_info[tb_key]['col'].append(article_info['index_row_id'])
                        time_bucket_inverted_ix_info[tb_key]['data'].append(article_info['index_row_id'])

    # Create embedding matrix.    
    for tb_key, tfidf_info in time_bucket_tfidf_info.items():
        _, bucket_language = tb_key
        vectorizer = vectorizer_sentpiece_en if bucket_language == LANG_EN else vectorizer_sentpiece_ru
        mx = scipy.sparse.csr_matrix((tfidf_info['data'], (tfidf_info['row'], tfidf_info['col'])),
            shape=(max(tfidf_info['row']) + 1, len(vectorizer.vocabulary_)))
        time_buckets_local[tb_key]['mx'] = mx

    # Create inverted index.
    for tb_key, tfidf_info in time_bucket_inverted_ix_info.items():
        _, bucket_language = tb_key
        vectorizer = vectorizer_sentpiece_en if bucket_language == LANG_EN else vectorizer_sentpiece_ru
        
        mx = scipy.sparse.csr_matrix((tfidf_info['data'], (tfidf_info['row'], tfidf_info['col'])), 
            shape=(len(vectorizer.vocabulary_), MAX_INDEX_SIZE)).tolil()
        time_buckets_local[tb_key]['inverted_ix'] = mx

    # Compute aggregate features for clusters.
    for (tb_key, _), cluster_info in clusters_local.items():
        cluster_info['max_time'] = max(cluster_info['article_times'])
        cluster_info['category'] = Counter(
            cluster_info['article_categories']).most_common(1)[0][0]

        # Sort articles by relevance.
        article_ids = cluster_info['article_ids'] 
        article_file_names = cluster_info['article_file_names'] 
        article_times = cluster_info['article_times'] 
        article_categories = cluster_info['article_categories'] 

        cluster_mx = time_buckets_local[tb_key]['mx'][article_ids, :]
        relevance_scores = (cluster_mx * cluster_mx.T).sum(axis=0).A[0]
        _, article_ids, article_file_names, article_times, article_categories = zip(*sorted(zip(
            relevance_scores, article_ids, article_file_names, article_times, article_categories), key=lambda x: -x[0]))

        cluster_info['article_ids'] = article_ids
        cluster_info['article_file_names'] = article_file_names
        cluster_info['article_times'] = article_times
        cluster_info['article_categories'] = article_categories

    all_articles = all_articles_local
    file_name_to_article = file_name_to_article_local
    max_publication_time = max_publication_time_local
    time_buckets = time_buckets_local


def get_response(command, source_dir):
    """Process a request."""

    # Parse files in the `source_dir`.
    article_info = get_article_info(source_dir)
    
    if command == 'languages':
        return get_languages_response(article_info)

    # Filter out non-{RU,EN} articles.
    article_info = [a for a in article_info if a['language'] in LANGUAGES]

    # Determine news category (used for both news filtering and classification).
    article_info = classify_news(article_info)

    # Filter out non-news articles.
    article_info = filter_news(article_info)
    if command == 'news':
        return get_filter_news_response(article_info)

    # Send news classification response.
    if command == 'categories':
        return get_classify_news_response(article_info)

    # Group articles into threads.
    article_threads = extract_threads(article_info)
    if command == 'threads':
        return get_threads_response(article_threads)

    # Extract top threads.
    top_threads = sort_threads(article_threads)
    if command == 'top':
        return top_threads


time_diff_sum = 1e-6
request_cnt = 1
init_time = time.time()
def timing_logger(prefix='Execution'):
    """Function execution timing decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global time_diff_sum, request_cnt
            start_time = int(time.time() * 1000)
            func(*args, **kwargs)
            end_time = int(time.time() * 1000)
            time_diff = end_time - start_time
            time_diff_sum += time_diff
            request_cnt += 1
            uptime_secs = int(time.time() - init_time) + 1
            logger.info(f'{prefix} {args[0].path} (Size:{args[0].headers["Content-Length"]}) done in {time_diff} ms ({start_time} >> {end_time})'
                        f' | Avg time: {int(time_diff_sum / request_cnt)} ms'
                        f' | R/Sec: {request_cnt / uptime_secs:.2f}'
                        f' | Rs: {request_cnt}'
                        f' | Ix size: {len(file_name_to_article)}')
        return wrapper
    return decorator


class RequestHandler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def check_state(self):
        """Respond with 503 if server is not ready to respond."""
        if CURRENT_SERVER_STATE != ServerState.READY:
            self.send_response(503)
            self.send_header('Content-Length', 0)
            self.end_headers()
            return False
        return True

    @timing_logger('GET')
    def do_GET(self):
        """Process GET request for getting news threads.

        GET /threads?period=<period>&lang_code=<lang_code>&category=<category> HTTP/1.1
        """
        if not self.check_state():
            return

        parsed_path = urlparse(self.path)

        # Parse query params.
        period = TTL_MAX
        lang_code = LANG_EN
        category = 'society'
        for param_name, param_value in parse_qsl(parsed_path.query):
            if param_name == 'period':
                period = int(param_value)
                period = max(min(period, TTL_MAX), TTL_MIN)
            elif param_name == 'lang_code' and param_value in LANGUAGES:
                lang_code = param_value
            elif param_name == 'category' and param_value in CATEGORIES | {'any'}:
                category = param_value

        threads = get_index_threads(period, lang_code, category)
        threads = json.dumps(threads, ensure_ascii=False).encode('utf-8', errors='ignore')

        self.send_response(200)
        self.send_header('Content-Type',
                         'application/json')
        self.send_header('Content-Length', len(threads))
        self.end_headers()
        self.wfile.write(threads)

    @timing_logger('PUT')
    def do_PUT(self):
        """Process PUT request with article content to be indexed.

        PUT /article.html HTTP/1.1
        Content-Type: text/html
        Cache-Control: max-age=<seconds>
        Content-Length: 9

        <content>
        """
        if not self.check_state():
            return

        with lock:
            file_name = urlparse(self.path).path.replace('/', '')
            article_exists = file_name in all_articles

            ttl_str = self.headers['Cache-Control'] or f'={TTL_NO_EXPIRATION}'
            ttl = max(min(int(ttl_str.split('=')[1]), TTL_MAX), TTL_MIN)
            length = int(self.headers['Content-Length'])
            article_content = self.rfile.read(length)
            article_content = article_content.decode('utf-8', 'ignore')[:MAX_CHARS]
            article_info = process_html(article_content, file_name)
            article_info['ttl'] = ttl

            skip_article = False
            if article_exists and (file_name in file_name_to_article):
                skip_article = all(article_info[f] == file_name_to_article[file_name][f] for f in ARTICLE_MUTABLE_FIELDS)

            if not skip_article:
                # In case if article exists and was updated remove it from existing clusters.
                if article_exists:
                    delete_article(file_name)

                # Add to the list of all articles.
                all_articles.add(file_name)

                # Add EN and RU news to the index.
                is_indexed = False
                if article_info['language'] in LANGUAGES:
                    article_info = classify_news([article_info])
                    article_info = filter_news(article_info)

                    if article_info:
                        article_info = article_info[0]
                        vectorizer = vectorizer_sentpiece_en if article_info['language'] == LANG_EN else vectorizer_sentpiece_ru
                        index_article(file_name, article_info, vectorizer)
                        is_indexed = True
                if not is_indexed:
                    ARTICLE_UPDATES_FILE.write({
                        'eventType': ArticleUpdate.IGNORE.name,
                        'file_name': file_name
                    })

            if article_exists:
                self.send_response(204)
            else:
                self.send_response(201)
            self.send_header('Content-Length', 0)
            self.end_headers()

    @timing_logger('DELETE')
    def do_DELETE(self):
        """Process DELETE request with article to delete from index.

        DELETE /article.html HTTP/1.1
        """
        if not self.check_state():
            return

        with lock:
            file_name = urlparse(self.path).path.replace('/', '')
            article_exists = file_name in all_articles

            if article_exists:
                delete_article(file_name)
                self.send_response(204)
            else:
                self.send_response(404)
            self.send_header('Content-Length', 0)
            self.end_headers()



if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('command', action='store', type=str)
    my_parser.add_argument('command_mod', action='store', type=str)
    args = my_parser.parse_args()

    # Load resources.
    if args.command in CLI_COMMANDS:
        if CURRENT_SERVER_STATE == ServerState.OFF:
            # Start the server.
            if args.command == 'server':
                server_port = int(args.command_mod)
                if not server_port:
                    raise Exception('Server port is not specified.')
                CURRENT_SERVER_STATE = ServerState.LOADING

                addr = ('localhost', server_port)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(addr)
                sock.listen()

                class Thread(threading.Thread):
                    def __init__(self, i):
                        threading.Thread.__init__(self)
                        self.i = i
                        self.daemon = True
                        self.start()
                    def run(self):
                        httpd = HTTPServer(addr, RequestHandler, False)
                        httpd.socket = sock
                        httpd.server_bind = self.server_close = lambda self: None
                        httpd.serve_forever()
                [Thread(i) for i in range(HTTP_SERVER_THREAD_CNT)]

                logger.info('Loading model files...')

            # Load language detection model.
            # https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
            lang_model = fasttext.load_model('resources/lid.176.bin')

            if args.command != 'languages':
                # Load PageRank data.
                with open('resources/pr.json', 'r') as f:
                    domain_pagerank = json.load(f)

                # Load token lists.
                with open('resources/tokens_ru.txt', 'r') as f:
                    ru_tokens = set([x.strip() for x in f.readlines()])
                with open('resources/tokens_en.txt', 'r') as f:
                    en_tokens = set([x.strip() for x in f.readlines()])
                with open('resources/tokens_blacklist_ru.txt', 'r') as f:
                    ru_tokens_blacklist = set([x.strip() for x in f.readlines()])
                with open('resources/tokens_blacklist_en.txt', 'r') as f:
                    en_tokens_blacklist = set([x.strip() for x in f.readlines()])
                tokens_all = {x for x in ru_tokens | en_tokens if x}
                tokens_blacklist_all = {x for x in ru_tokens_blacklist | en_tokens_blacklist if x}

                # Load word normalisation lists.
                with open('resources/word2norm_ru.pickle', 'rb') as f:
                    word2norm_ru = pickle.load(f)
                with open('resources/word2norm_en.pickle', 'rb') as f:
                    word2norm_en = pickle.load(f)
                word2norm = word2norm_ru
                word2norm.update(word2norm_en)

                # Load news classifiers.
                with open('resources/clf_en.pickle', 'rb') as f:
                    clf_en = pickle.load(f)
                with open('resources/clf_ru.pickle', 'rb') as f:
                    clf_ru = pickle.load(f)

                # Load SentencePiece models.
                sp_en = spm.SentencePieceProcessor()
                sp_en.load('resources/sentpiece_en_10k.model')
                sp_ru = spm.SentencePieceProcessor()
                sp_ru.load('resources/sentpiece_ru_10k.model')

                # Load TF-IDF vectorizers.
                with open('resources/vectorizer_en.pickle', 'rb') as f:
                    vectorizer_en = pickle.load(f)
                with open('resources/vectorizer_ru.pickle', 'rb') as f:
                    vectorizer_ru = pickle.load(f)
                with open('resources/vectorizer_sentpiece_en.pickle', 'rb') as f:
                    vectorizer_sentpiece_en = pickle.load(f)
                    vectorizer_sentpiece_en.tokenizer = sp_en.encode_as_ids
                with open('resources/vectorizer_sentpiece_ru.pickle', 'rb') as f:
                    vectorizer_sentpiece_ru = pickle.load(f)
                    vectorizer_sentpiece_ru.tokenizer = sp_ru.encode_as_ids

                # Load boosted tokens which will have extra weight in clustering.
                with open('resources/boosted_tokens.txt', 'r') as f:
                    boosted_tokens = set([x.strip() for x in f.readlines()])
                with open('resources/sentpiece_ru_10k.vocab', 'r') as f:
                    sp_vocab_ru = [x.strip().split('\t')[0].replace('_', '') for x in f.readlines()]
                boosted_token_ids = set()
                for idx, token in enumerate(sp_vocab_ru):
                    for candidate_token in boosted_tokens:
                        if '▁' + candidate_token == token:
                            boosted_token_ids.add(idx)

                if args.command == 'server':
                    logger.info('Initialising index...')
                    t = time.time()
                    maybe_load_from_disk()
                    logger.info(f'Loaded {len(all_articles)} articles')
                    logger.info(f'Index size: {len(file_name_to_article)}')
                    logger.info(f'Index initialisation completed in {time.time() - t:.2f} sec')

                CURRENT_SERVER_STATE = ServerState.READY
            else:
                word2norm = {}
                domain_pagerank = {}

            # Run in CLI mode.
            if args.command != 'server':
                result = get_response(args.command, Path(args.command_mod))
                print(json.dumps(result, ensure_ascii=False))
            else:
                time.sleep(9e7)

    else:
        raise Exception('Unknown command, please use one of: languages, news, categories, threads, server.')
