# Search_Engine_wikipedia
Wikipedia Search Engine for Data Retrieval Course:
#Introduction in this project we create a search engine for a wikipedia corpus.
at the beginning we aused google colab to create to create the inverted_index_gcp that will be the core of our engine.
we used preproccesed data (indexes and dictionaries) so our search engine could access all data in a fast way.
each data source the this engine is using is calculted with a diffrent weight to give us a more precision for our final most relevent results.

code structure and organization:

### **inverted_index_gcp**: 
this file is the base of our engine and is a class that we use to create and use our indexes

### search_frontend: 
this is the file that runs the engine. in this file we use all our data and methods to perform evaluations that when given a query will help us return the most releveant wikipedia pages.

### **Methods**:

### search_title:
this method tokenize the query and then itartes over the query and creating the posting list thanks to the inverted title index,
return the best documents based on title name.

### search_anchor:
this method tokenize the query and then itartes over the query and creating the posting list thanks to the inverted anchor index,
return the best documents based on anchor.

### search_body:
this method tokenize the query and then itartes over the query and creating the posting list thanks to the inverted body index,
return the 100 best documents based on body text.

### get_pageview:
his method iterates over the id's and with the page view dictionary it returns the pairing page view for all the terms in the query.

### get_pagerank:
his method iterates over the id's and with the page view dictionary it returns the pairing page view for all the terms in the query.

### search:
this is the main function and the most recommemded to use, this method uses all the methods above and a few methods to help to calculate the final score for each page based on the query.
this method gives weight for each result from the other methods so we could get the best overall reuslts.
