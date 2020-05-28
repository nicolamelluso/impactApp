# coding: utf-8
"""
Example of a Streamlit app for an interactive spaCy model visualizer. You can
either download the script, or point streamlit run to the raw URL of this
file. For more details, see https://streamlit.io.
Installation:
pip install streamlit
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download de_core_news_sm
Usage:
streamlit run streamlit_spacy.py
"""
from __future__ import unicode_literals

import graphbrain
import streamlit as st
import base64
import os
import pandas as pd
import numpy as np

import hypergraphs as hg

#import graphbrain

from graphbrain.parsers import *
from graphbrain import notebook

@st.cache(allow_output_mutation=True)
def load_parser():
    return create_parser(name='en')

@st.cache(allow_output_mutation=True)
def load_app_sentences():
    return pd.read_pickle('./data/app_sentences.pickle')

@st.cache(allow_output_mutation=True)
def load_sents():
    return pd.read_pickle('./data/sents.pickle')

sents = load_sents()
parser = load_parser()
dfs = load_app_sentences()


import re

def adjust(x):
    
    if x['trigger'] == 'to':
        if x['predicate'] == 'lead':
            x['predicate'] = 'lead to'
            x['trigger'] = np.nan
            
    if x['entity'].startswith('across'):
        x['entity'] = x['entity'].replace('across','')
        
        x['trigger'] = 'across'
    
    return x

def tag_concept(x):
    
    for edge in x['edge'].subedges():
        if 'cp' in edge.type():
            return 'Agent'
    else:
        return 'Topic'
    
def tag_verb(x):
    if bool(re.search('adopt|identi|develop|show|achiev|asses|creat|buil|ensur|establish|evaluat|indentif|improv|increas|inform|produc|provid|shap|support|understand',x['predicate'])):
        return 'Indirect Impact'
    elif bool(re.search('lead|led|use|help|impact|contribute',x['predicate'])):
        return 'Direct Impact'
    else:
        return '-'
    
def transform(x):
    if x['arg'] == 's':
        return 'subject'
    elif x['arg'] == 'o':
        return 'object'
    elif x['arg'] == 'x':
        return 'specification'
    else:
        return x['arg']


def process_edge(sent):
    
    try:
        sent = [s for s in sent['main_edge'].subedges() if s.is_atom() == False]
    
    
        sentences = []
        for edge in sent:
            if edge is not None:
                edge_verbs = []
                for edge_verb in hg.split(edge):
                    edge_verbs.extend(hg.edge_split(edge_verb))
                sentences.append(edge_verbs)
                
    
    
        sentences = pd.concat([pd.DataFrame(s) for s in sentences]).drop_duplicates()[['predicate','arg','entity','eID']].dropna()
        sentences['predicate'] = sentences.apply(lambda x: x['predicate'].label(), axis = 1)
        
        sentences = sentences[['predicate','arg','entity']].drop_duplicates()
        sentences['edge'] = sentences['entity']
        sentences['entity'] = sentences['entity'].apply(lambda x: x.label() if not pd.isnull(x) else x)
        
        
        sentences['trigger'] = sentences.apply(lambda x: x['arg'].label() if type(x['arg']) is graphbrain.hyperedge.Atom else np.nan, axis = 1)
        
        sentences = sentences[['edge','predicate','arg','trigger','entity']].drop_duplicates()
        sentences['arg'] = sentences.apply(lambda x: 'x' if type(x['arg']) is graphbrain.hyperedge.Atom else x['arg'], axis = 1)
        sentences = sentences.apply(lambda x: adjust(x),axis = 1)
        sentences['concept_type'] = sentences.apply(lambda x: tag_concept(x), axis = 1)
        sentences['verb_type'] = sentences.apply(lambda x: tag_verb(x), axis = 1)
        sentences = sentences[['verb_type','predicate','arg','trigger','entity','concept_type']]
        sentences['arg'] = sentences.apply(lambda x: transform(x), axis = 1)
        sentences = sentences.fillna('-')
        
        if ('Direct Impact' in sentences['verb_type'].unique()) | ('Indirect Impact' in sentences['verb_type'].unique()):
            return sentences
        else:
            return sentences[['predicate','arg','trigger','entity']]
        
    except Exception:
        return None



PAGES = [
    "Explore Classification",
    "Try to classify"
]

st.sidebar.markdown("# Examining societal impact of research with Semantic Hypergraphs")
st.sidebar.markdown("## Navigation")
selection = st.sidebar.radio("Go to", list(PAGES))

#href = f'<a href="https://github.com/nicolamelluso/impactApp/blob/master/data/sents_ALL.xlsx" download="impact_database.xlsx">Download the Database here</a>'
st.sidebar.markdown('[https://github.com/nicolamelluso/impactApp/blob/master/data/sents_ALL.xlsx](Download the databse here)', unsafe_allow_html=True)



if selection == 'Try to classify':

    DEFAULT_TEXT = "The research at the University of Glasgow lead to improvement of access to justice for victims of domestic abuse, shaping legislative change and providing the resources required for practitioners in this field to identify best practices across Scotland."

    HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 0.5rem; margin-bottom: 0.5rem">{}</div>"""

    
    
    st.header("Try to classify your sentence")
    
    st.markdown('This demo helps you to understand how the classification of impact sentence works.')
    st.markdown('This tool uses the Semantic Hypergraphs. See the full documentation [here](http://graphbrain.net/reference/notation.html)')
    
    text = st.text_area("Text to analyze", DEFAULT_TEXT)
    sent = parser.parse(text)[0]
    
    df = process_edge(sent)
    
    
    
    sent = [s for s in sent['main_edge'].subedges() if s.is_atom() == False]
    
    
    if df is None:
        st.warning('This is not an impact sentence')
    elif 'concept_type' not in df.columns:
        st.warning('This is not an impact sentence')
    else:
        st.info('This is impact sentence')
    
        ### SHOW THE TABLE
        st.table(df)
        
        ### SHOW THE HYPERGRAPH
        html = ''
        for edge in sent:
        
            html += '<p>'
            html += notebook._edge2html(edge, roots_only=False, formatting='oneline')
            html += '</p>'
    
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    
        
        
    
    
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="hyperedges.csv">Download hyperedges</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

else:
    
    st.header('Examining societal impact of research with Semantic Hypergraphs')
    st.markdown('In this page it is possible to explore a dump of random sentences from the analysis of impact sentences')

    HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 0.5rem; margin-bottom: 0.5rem">{}</div>"""
    
    DEFAULT_SENT = '1000S00004'
    sent_selection = st.radio("What kind of sentence do you want to explore?", ['Impact Sentence','Non-Impact Sentence'])
    
    sents = sents[:50]
    shows = sents[sents['type'] == sent_selection]
    sent = st.selectbox('Choose the sentence',[s[:50] + '...' for s in shows['sent'].unique().tolist()])
        
#    sentId = st.text_area("Sentence to analize", DEFAULT_SENT)
    
    sent = dfs[dfs['sent'].str.contains(sent[:-3])]['sent'].unique()[0]
    explore = dfs[dfs['sent'].str.contains(sent[:-3])].drop(['sentId','sent'],axis = 1)

    st.write(HTML_WRAPPER.format(sent), unsafe_allow_html=True)
    st.table(explore[['verb_type','predicate','arg','trigger','entity','concept_type']].fillna('-'))
    