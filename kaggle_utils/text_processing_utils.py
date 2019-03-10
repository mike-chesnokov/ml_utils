# Utils for text processing
# Methods "text_process_mystem", "text_process_pymorph", "text_process_nltk_stem"
# can be used within pandas apply:
# text_field.apply(lambda x: method(x))

import re
from string import punctuation

import numpy as np
import pandas as pd
from pymystem3 import Mystem
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stopwords_rus = stopwords.words("russian")

morph = MorphAnalyzer()  # pymorphy
mystem = Mystem()  # mystem
stemmer = SnowballStemmer('russian')  # nltk stemmer

pattern1 = re.compile(r'[^а-яa-z ]')  # for deleting non alphabetic
pattern2 = re.compile(r'\b\w\b')  # for deleting one symbol words
pattern3 = re.compile(r'\s+')  # for deleting double spaces


def text_preprocess(text):
    """
    Method for russian text preprocessing:
    - replace popular symbols
    - delete non alphabetic symbols
    - delete one symbol words
    - delete double spaces

    :param text: string to process
    :return: processed string
    """
    temp_0 = text.lower() \
        .replace('wi-fi', 'wifi') \
        .replace('ё', 'е')

    temp1 = re.sub(pattern1, ' ', temp_0)
    temp2 = re.sub(pattern2, '', temp1)
    temp3 = re.sub(pattern3, ' ', temp2).strip()
    return temp3


def text_process_mystem(text):
    """
    Method for text lemmatization (get normal form of word) with Mystem

    :param text: string to process
    :return: processed string
    """
    temp1 = text_preprocess(text)
    temp2 = [word for word in mystem.lemmatize(temp1) if word != ' ' and word != '\n']
    return ' '.join(temp2)


def text_process_pymorph(text):
    """
    Method for text lemmatization (get normal form of word) with Pymorphy

    :param text: string to process
    :return: processed string
    """
    temp1 = text_preprocess(text)
    temp2 = [morph.parse(word)[0].normal_form for word in temp1.split()]
    return ' '.join(temp2)


def text_process_nltk_stem(text):
    """
    Method for text stemming (cut word to main part) with NLTK

    :param text: string to process
    :return: processed string
    """
    temp1 = text_preprocess(text)
    temp2 = [stemmer.stem(word) for word in temp1.split()]
    return ' '.join(temp2)


pattern_digit = re.compile(r'\d+')  # for digits count
pattern_year = re.compile(r'\d\d\d\d')  # for year count
pattern_latin = re.compile(r'[a-z]')  # for latin chars count


def make_text_features(df, text_col, log_num=False):
    """
    Make several features from russian text

    :param df: pandas dataframe with features
    :param text_col: text col to process
    :param log_num: to use log transform after feature created
    :return: df_: new dataframe with features
    """
    df_ = df.copy()

    if not log_num:
        df_[text_col + '_num_chars'] = df_[text_col].apply(len)
        df_[text_col + '_num_words'] = df_[text_col].apply(lambda x: len(x.split()))
        df_[text_col + '_num_punct'] = df_[text_col].apply(lambda x: len([char for char in x if char in punctuation]))
        df_[text_col + '_cnt_digit'] = df_[text_col].apply(lambda x: len(re.findall(pattern_digit, x)))
        df_[text_col + '_cnt_latin'] = df_[text_col].apply(lambda x: len(re.findall(pattern_latin, x)))

    else:
        df_[text_col + '_num_chars'] = df_[text_col].apply(len).apply(np.log1p)
        df_[text_col + '_num_words'] = df_[text_col].apply(lambda x: len(x.split())) \
            .apply(np.log1p)
        df_[text_col + '_num_punct'] = df_[text_col].apply(lambda x: len([char for char in x if char in punctuation])) \
            .apply(np.log1p)
        df_[text_col + '_cnt_digit'] = df_[text_col].apply(lambda x: len(re.findall(pattern_digit, x))) \
            .apply(np.log1p)
        df_[text_col + '_cnt_latin'] = df_[text_col].apply(lambda x: len(re.findall(pattern_latin, x))) \
            .apply(np.log1p)

    # common features
    df_[text_col + '_words_len'] = df_[text_col + '_num_chars'] / (1 + df_[text_col + '_num_words'])
    df_[text_col + '_has_year'] = df_[text_col].apply(lambda x: 1 if re.search(pattern_year, x) else 0)
    df_[text_col + '_start_digit'] = df_[text_col].apply(lambda x: x[0].isdigit()).astype(int)
    df_[text_col + '_end_digit'] = df_[text_col].apply(lambda x: x[-1].isdigit()).astype(int)
    # count abbreviations
    df_[text_col + '_cnt_bytes'] = df_[text_col].apply(lambda x:
                                                       x.count('mb') + x.count('gb') + x.count('tb') +
                                                       x.count('мб') + x.count('гб') + x.count('тб')
                                                       )
    df_[text_col + '_cnt_sizes'] = df_[text_col].apply(lambda x: x.count('мм') + x.count('см'))
    df_[text_col + '_cnt_weight'] = df_[text_col].apply(lambda x: x.count('гр') + x.count('кг'))
    df_[text_col + '_cnt_times'] = df_[text_col].apply(lambda x: x.count('сек') + x.count('мин') + x.count('час'))
    # count man, woman and child
    df_[text_col + '_cnt_man'] = df_[text_col].apply(lambda x: x.count(' муж'))
    df_[text_col + '_cnt_woman'] = df_[text_col].apply(lambda x: x.count(' жен'))
    df_[text_col + '_cnt_child'] = df_[text_col].apply(lambda x:
                                                       x.count(' дет') + x.count(' реб') +
                                                       x.count(' мальч') + x.count(' девоч')
                                                       )
    return df_
