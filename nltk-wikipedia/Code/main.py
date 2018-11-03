    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:32:33 2018

@author: jinyubo
"""
import wikipedia
import midterm_functions as mf
from gensim.summarization.summarizer import summarize
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/<name>', methods=['GET'])
def project(name):
    
    text = wikipedia.page(name)
    content = text.content
    summary = mf.get_summary(name)
    keywords = mf.get_keywords(content)
    abstract = summarize(content,ratio=0.05)
    pos_labels = mf.get_PosNegWords(abstract)[0]
    pos_labels = ', '.join(pos_labels)
    neg_labels = mf.get_PosNegWords(abstract)[1]
    neg_labels = ', '.join(neg_labels)
    
    return render_template('midterm.html', title = text.title, 
                           summary = summary, keywords = keywords,
                           positive_words = pos_labels,
                           negative_words = neg_labels)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True) 
    