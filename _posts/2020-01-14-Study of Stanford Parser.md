---
layout:     post
title:      A brief study of Stanford Parser in Stanford NLP
date:       2020-01-14
author:     Yongchao Wu
header-img: img/home-bg-tv.jpg
catalog: true

---

## Introduction

Stanford CoreNLP offers a collection of computational resources for the processing of natural language. Many NLP tools can be found in Stanford CoreNLP, 
such as the part-of-speech(POS) tagger, Stanford parser and sentiment analysis. This survey will focus on the Stanford parser which can accomplish different multilingual parsing tasks. 
To understand how Stanford parsers can support multilingual parsing with good performance, papers about the Universal Dependencies (UD) framework,  Neural Networks based parser and Compositional Vector Grammar parser will be studied. 


## Universal Dependencies

>Key Word：universal Dependency


Multilingual syntax and parsing work have long been impeded by the fact that annotation schemes differ widely across languages. The Universal Dependencies (UD) project aims to address this problem by creating cross-linguistically compatible treebank annotations for many languages. 


### Annotation Principles

The UD annotation is based on dependency and lexicalism aiming at capturing the syntactic relations of the words with morphological properties, the basic principle of which is to create a multilingual annotation framework that enables the same structure of sentences in different languages to be annotated in the same way consistently and transparently.

#### Word Segmentation

Words with spaces do not count as single tokens in UD. In this case, UD will annotate the expressions with special dependency relations. 

#### Morphology

In UD, three layers of knowledge can be found from the perspective of word morphology, they are a lemma, a POS tag, and a set of features containing lexical and grammatical information. The POS tags are divided into three categories, open class words, closed class words, and others. There are two types of morphological features, lexical features, and inflectional features.

#### Syntax

UD describes the relationships of dependency between words from four points of view, grammatical relationships, relationships between words of content, enhanced representations, and linguistic relationships.

##### Grammatical Relations

According to the relationship of three structures (nominals, clauses and modifier words),  different dependency relationships between words are organized. The initiative also distinguishes from other dependents between core arguments (e.g. subject and object). The UD can catch anomalies that occur in unedited or informal texts as well as identifing word relationships in compounding cases.

##### Relations between Content Words

When defining dependency relations, content words are given high priority, that means the content words are directly related through dependency relations, the function words are regarded as attachments to the content words they describe, and punctuations are also regarded as attachments to the "root".

##### Enhanced Representation

UD adoptes enhanced dependency representation as improvement of basic dependencies.



