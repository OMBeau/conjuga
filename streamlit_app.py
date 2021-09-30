from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google_trans_new import google_translator
from googletrans import Translator

show_spinner = False

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"
}

rename_conjuga = {
    ("Indicativo", "Pretérito imperfeito"): ("Indicativo", "Pretérito Imperfeito"),
    ("Indicativo", "Pret. mais-que-perfeito"): (
        "Indicativo",
        "Pretérito Mais-que-Perfeito",
    ),
    ("Indicativo", "Futuro /Futuro do presente"): (
        "Indicativo",
        "Futuro do Presente Simples",
    ),
    ("Indicativo", "CONDICIONAL /Futuro do pretérito"): (
        "Condicional",
        "Futuro do Pretérito Simples",
    ),
    ("Conjuntivo / Subjuntivo", "Pretérito imperfeito"): (
        "Conjuntivo / Subjuntivo",
        "Pretérito Imperfeito",
    ),
    ("Conjuntivo / Subjuntivo", "Pretérito imperfeito"): (
        "Conjuntivo / Subjuntivo",
        "Pretérito Imperfeito",
    ),
    ("Infinitivo Pessoal", None): ("Infinitivo", "Pessoal"),
}

rename_reverso = {
    ("Infinitivo", ""): ("Infinitivo", "Pessoal"),
    ("Imperativo", ""): ("Imperativo", "Afirmativo"),
    ("Imperativo Negativo", ""): ("Imperativo", "Negativo"),
}


def rename_d_indexes(verb_dict, rename_dict):
    verb_d_new = {}
    for k0, v0 in verb_dict.items():
        for k1, v1 in v0.items():
            if (k0, k1) in rename_dict:
                replace_val = rename_dict[(k0, k1)]
                if replace_val[0] not in verb_d_new:
                    verb_d_new[replace_val[0]] = {}
                verb_d_new[replace_val[0]][replace_val[1]] = v1
            else:
                if k0 not in verb_d_new:
                    verb_d_new[k0] = {}
                verb_d_new[k0][k1] = v1

    return verb_d_new


@st.cache(hash_funcs={BeautifulSoup: lambda _: None}, show_spinner=show_spinner)
def conjuga_me(verb):
    url = f"https://conjuga-me.net/verbo-{verb}"
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.content, "html.parser")


@st.cache(hash_funcs={BeautifulSoup: lambda _: None}, show_spinner=show_spinner)
def conjuga_reverso(verb):
    url = f"https://conjugator.reverso.net/conjugation-portuguese-verb-{verb}.html"
    response = requests.get(url)
    return BeautifulSoup(response.content, "html.parser")


def contexto_reverso(verb):
    url = f"https://context.reverso.net/translation/portuguese-english/{verb}"
    response = requests.get(url)
    return BeautifulSoup(response.content, "html.parser")


def synonyms_reverso(verb):
    pass


@st.cache(show_spinner=show_spinner)
def get_target_verb(verb):
    soup = conjuga_reverso(verb)
    if target_verb := soup.find(id="ch_lblVerb"):
        return target_verb.text
    else:
        return None


def english_meanings(verb):
    # Banned IP
    soup = contexto_reverso(verb)
    if meanings := soup.find(id="translations-content"):
        return meanings.text
    else:
        return None
    # for m in meanings


def english_meanings_short(verb):
    soup = conjuga_reverso(verb)
    if meanings := soup.find(id="list-translations"):
        return meanings
    else:
        return None


@st.cache(hash_funcs={Translator: lambda _: None}, show_spinner=show_spinner)
def get_translator():
    return Translator()


@st.cache(hash_funcs={Translator: lambda _: None}, show_spinner=show_spinner)
def pt_to_en_goog(text, translator):
    return translator.translate(text, src="pt", dest="en").text


@st.cache(hash_funcs={Translator: lambda _: None}, show_spinner=show_spinner)
def en_to_pt_goog(translator, text_en):
    return translator.translate(text_en, src="en", dest="pt").text


def pt_to_en_new(translator, text):
    # t = google_translator(timeout=5)
    return translator.translate(text, lang_src="pt", lang_tgt="en")


@st.cache(show_spinner=show_spinner)
def multi_pt_to_en(pt_conjugations):
    translator_new = google_translator(timeout=5)
    pool = ThreadPool(8)  # Threads
    try:
        english_vals = pool.map(partial(pt_to_en_new, translator_new), pt_conjugations)
    except Exception as e:
        raise e
    pool.close()
    pool.join()
    return english_vals


def get_verb_dict_conjuga(soup):  # sourcery no-metrics

    soup_mobile = soup.find(id="main-table-mobile")
    d = {}
    tense_index = 0
    for i, div in enumerate(soup_mobile.find_all("div")):
        if div.get("style"):
            continue

        # Establish Modo
        if div["class"][0] == "modo-mobile":
            modo = div.text.title()
            d[modo] = {}

        # Establish Tense
        if div["class"][0].startswith("tempo-mobile"):
            if i - 1 != tense_index:
                tense = div.text
            else:
                del d[modo][tense]
                tense += div.text

            if tense == "\xa0":
                tense = None

            d[modo][tense] = {}
            tense_index = i

        if div["class"][0] == "pronome":
            pronoun = div.text.strip()
            prefix = None
            suffix = None
            if modo == "Conjuntivo / Subjuntivo":
                prefix, pronoun = pronoun.split()
                if tense in ["Pretérito imperfeito", "Futuro"]:
                    prefix = "que/se/quando"

        if div["class"][0].startswith("conjugation"):
            conjugation = div.text
            if modo in ["Imperativo", "Infinitivo Pessoal"]:
                parts = conjugation.strip().split()
                prefix = None
                suffix = None
                if len(parts) == 1:
                    conjugation = parts[0]
                    pronoun = "eu"
                elif len(parts) == 2:  # IMPERATIVO-Afirmativo
                    conjugation, suffix = parts
                elif len(parts) == 3:  # IMPERATIVO-Negativo or INFINITIVO PESSOAL
                    prefix, conjugation, suffix = parts

                if suffix:
                    pronoun = suffix.replace("(", "").replace(")", "")

            irregular = bool(div.find("irreg"))

            # prefix = prefix or ""
            # suffix = suffix or ""

            if pronoun == "ele/ela":
                pronoun += "/você"
            elif pronoun == "eles/elas":
                pronoun += "/vocês"

            d[modo][tense][pronoun] = {
                "prefix": prefix,
                "conjugation": conjugation,
                "irregular": irregular,
                "suffix": suffix,
                "full_conjugation": " ".join(
                    [prefix or "", conjugation, suffix or ""]
                ).strip(),
            }

    return d


def get_verb_dict_reverso(soup):  # sourcery no-metrics

    soup = soup.find(class_="result-block-api")
    d = {}
    pronouns = ["eu", "tu", "ele/ela/você", "nós", "vós", "eles/elas/vocês"]
    modo = ""
    for i, div in enumerate(soup.find_all("div")):

        if div["class"][0] == "word-wrap-title":
            modo = div.text.strip()

            if modo not in ["Particípio", "Gerúndio"]:
                d[modo] = {}

        if modo in ["Particípio", "Gerúndio"]:
            continue

        if div["class"][0] == "blue-box-wrap":

            # get tense
            if tense := div.find("p"):
                tense = tense.text
            else:
                tense = ""
            d[modo][tense] = {}

            for p_idx, p in enumerate(div.find_all("li")):

                # get pronoun
                if pronoun := p.find(class_="graytxt"):
                    pronoun = pronoun.text
                else:
                    pronoun = ""

                # get conjugation
                if conjugation := p.find(h="1"):
                    conjugation = conjugation.text
                else:
                    conjugation = ""

                # get full_conjugation
                i = p.find("i")
                full_conjugation = [i.text]
                for j in i.find_next_siblings("i"):
                    full_conjugation.append(j.text)
                full_conjugation = " ".join(full_conjugation)

                if modo in ["Infinitivo", "Imperativo", "Imperativo Negativo"]:
                    pronoun = pronouns[p_idx]
                    full_conjugation += f" ({pronoun})"

                # found_verb boolean
                found_verb = False
                if p.find(class_="hglhOver"):
                    found_verb = True

                d[modo][tense][pronoun] = {
                    "found_verb": found_verb,
                    "conjugation": conjugation,
                    "full_conjugation": full_conjugation,
                }
    return d


def flatten_dict(d):
    reformed_dict = {}
    for key0, dict0 in d.items():
        for key1, dict1 in dict0.items():
            for key2, values in dict1.items():
                reformed_dict[(key0, key1, key2)] = values

    return reformed_dict


def verb_to_df(verb_dict, rename_dict):
    verb_dict_renamed = rename_d_indexes(verb_dict, rename_dict)
    verb_dict_renamed_flat = flatten_dict(verb_dict_renamed)
    return (
        pd.DataFrame(verb_dict_renamed_flat)
        .T.reset_index()
        .rename(columns={"level_0": "modo", "level_1": "tense", "level_2": "pronoun"})
    )


@st.cache(show_spinner=show_spinner)
def process_table_conjuga(target_verb):
    soup_conjuga = conjuga_me(target_verb)
    verb_dict_conjuga = get_verb_dict_conjuga(soup_conjuga)
    df_conjuga = verb_to_df(verb_dict_conjuga, rename_conjuga)
    return df_conjuga.drop(columns=["prefix", "suffix"])


@st.cache(show_spinner=show_spinner)
def process_table_reverso(verb):
    soup_reverso = conjuga_reverso(verb)
    verb_dict_reverso = get_verb_dict_reverso(soup_reverso)
    return verb_to_df(verb_dict_reverso, rename_reverso)


@st.cache(show_spinner=show_spinner)
def process_table_combined(df_conjuga, df_reverso):
    return df_reverso.merge(
        df_conjuga,
        how="left",
        left_on=["modo", "tense", "pronoun"],
        right_on=["modo", "tense", "pronoun"],
        suffixes=("_reverso", "_conjuga"),
    )


def remove_prefix(row):
    c1 = row["full_conjugation_reverso"]
    c2 = row["pronoun"]
    if c1.startswith(c2):
        return c1.split(" ", 1)[1]
    else:
        return c1


def sidebar_selections(df, col):
    st.sidebar.markdown(f"## Select {col.title()}")
    col = col.lower()
    options = df[col.lower()].unique()
    selected = [m for m in options if st.sidebar.checkbox(m, key=f"{col}-{m}")]
    if selected:
        df = df[df[col].isin(selected)]
    return df


def style_irreg(v, mask):
    return mask.replace(True, "color: salmon")


def style_irreg_unknown(v, mask):
    return mask.replace(True, "background-color: lightgrey")


def style_found(v, mask):
    return mask.replace(True, "background-color: lavender")


def style_table(df, to_english=False):

    # Split table to get found_verb and irregular booleen masks
    val_types = df.columns.get_level_values(0)
    df_found = df.iloc[:, val_types == "found_verb"].droplevel(0, axis=1).fillna(False)
    df_irreg = df.iloc[:, val_types == "irregular"].droplevel(0, axis=1).fillna(False)
    df_irreg_unknown = df.iloc[:, val_types == "irregular"].droplevel(0, axis=1).isna()

    col = "english" if to_english else "full_conjugation_reverso"
    df = df.iloc[:, val_types == col].droplevel(0, axis=1).fillna("")

    # # Add Style to found verbs
    return (
        df.style.apply(style_irreg, mask=df_irreg, axis=None)
        .apply(style_irreg_unknown, mask=df_irreg_unknown, axis=None)
        .apply(style_found, mask=df_found, axis=None)
    )


def table_html_with_soup(df):

    soup = BeautifulSoup(df.to_html(), "html.parser")

    # work with just the Table header.
    thead = soup.find("thead")

    # iterate through header rows
    for tr in thead.find_all("tr"):
        # Remove text ("modo", "tense", "pronoun")
        idx_name = tr.find("th").string.extract()
        if idx_name == "pronoun":
            # Remove Entire table row
            tr.decompose()

    # Need formatter to preserve &nbsp;
    return str(soup.prettify(formatter="html"))


def main():

    verb = st.text_input("Verb in PT").strip()

    target_verb = get_target_verb(verb)

    if not target_verb:
        st.error("Not valid Portugese verb form.")
        return

    translator_goog = get_translator()

    meaning = pt_to_en_goog(target_verb, translator_goog)
    st.markdown(f"### Root Verb: **{target_verb}**  ({meaning})")

    text_en = st.sidebar.text_input("English to Portugese")
    if text_en:
        en_to_pt = en_to_pt_goog(translator_goog, text_en)
        st.sidebar.write(en_to_pt)

    # Prep Conjuga Table
    df_conjuga = process_table_conjuga(target_verb)

    # Prep Reverso Table
    df_reverso = process_table_reverso(verb)

    # Combine Tables
    df = process_table_combined(
        df_conjuga, df_reverso
    ).copy()  # copy to prevent cache mutation

    # Sidebar - Select Modo
    df = sidebar_selections(df, "modo")

    # Sidebar - Select Tense
    df = sidebar_selections(df, "tense")

    # Checkbox - English Tranlsation
    to_english = st.checkbox("English Translation")
    df["english"] = pd.Series(dtype=object)
    if to_english:
        pt_conjugations = df["full_conjugation_reverso"].to_list()
        df["english"] = multi_pt_to_en(pt_conjugations)

    # Checkbox - Verb with Prefix
    with_prefix = st.checkbox("Verb with Prefix")
    if not with_prefix:
        df["full_conjugation_reverso"] = df.apply(remove_prefix, axis=1)

    # Pivot table to show verb, irregular bool, and found_verb bool
    df = df.pivot(
        index="pronoun",
        columns=["modo", "tense"],
        values=[
            "full_conjugation_reverso",
            "english",
            "irregular",
            "found_verb",
        ],
    ).reindex(index=["eu", "tu", "ele/ela/você", "nós", "vós", "eles/elas/vocês"])

    if st.checkbox('Drop "vós"'):
        df = df.drop("vós")

    # Checkbox - Transpose
    if st.checkbox("Transpose"):

        idx = pd.IndexSlice
        for c in df.droplevel(0, axis=1).columns.unique():
            # ie c = ("Indicativo", "Presente")

            dff = df.loc[:, idx[:, c[0], c[1]]]
            col = dff.columns[0]
            st.markdown(f"### {col[1]} - {col[2]}")

            dff_style = style_table(dff)

            if with_prefix:
                dff_style = dff_style.hide_index()

            dff_html = dff_style.hide_columns().to_html()

            if to_english:
                dff_en = style_table(dff, to_english)
                dff_en = dff_en.hide_index().hide_columns()

                dff_style = dff_style.set_table_attributes("style='display:inline'")
                dff_en = dff_en.set_table_attributes("style='display:inline'")

                dff_html = dff_style.to_html() + dff_en.to_html()

            st.markdown(
                dff_html,
                unsafe_allow_html=True,
            )
    else:
        df = style_table(df, to_english)

        if with_prefix:
            df = df.hide_index()
            df_html = df.to_html()
        else:
            df_html = table_html_with_soup(df)

        st.markdown(
            df_html,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Conjuga", layout="wide", initial_sidebar_state="expanded"
    )
    main()
