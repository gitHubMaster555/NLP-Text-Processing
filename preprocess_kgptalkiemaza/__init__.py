from preprocess_kgptalkiemaza import utils


def get_word_counts(x):
    return utils._get_word_counts(x)


def get_charcounts(x):
    return utils._get_charcounts(x)


def get_avg_wordlength(x):
    return utils._get_avg_wordlength(x)


def get_stopword_counts(x):
    return utils._get_stopword_counts(x)


def get_hashtag_counts(x):
    return utils._get_hashtag_counts(x)


def get_mentions_counts(x):
    return utils._get_mentions_counts(x)


def get_digit_counts(x):
    return utils._get_digit_counts(x)


def get_uppercase_counts(x):
    return utils._get_uppercase_counts(x)


def get_cont_counts(x):
    return utils._get_cont_counts(x)


def get_emails(x):
    return utils._get_emails(x)


def remove_emails(x):
    return utils._remove_emails(x)


def get_urls(x):
    return utils._get_urls(x)


def remove_urls(x):
    return utils._remove_urls(x)


def remove_rt(x):
    return utils._remove_rt(x)


def remove_special_chars(x):
    return utils._remove_special_chars(x)


def remove_html_tags(x):
    return utils._remove_html_tags(x)


def remove_accents(x):
    return utils._remove_accents(x)


def remove_stopwords(x):
    return utils._remove_stopwords(x)


def make_base(x):
    return utils.make_base(x)


def remove_common_words(x, n=20):
    return utils._remove_common_words(x, n=20)


def remove_rare_words(x, n=20):
    return utils._remove_rare_words(x, n=20)


def spelling_correction(x):
    return utils._spelling_correction(x)