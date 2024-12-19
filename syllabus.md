# Building GenAI Apps Learning Badge Path
These are my notes as I re complete the  MongoDB learning path on building RAG applications.  I may create myself a project out of it as well, but one step at a time.


see: https://learn.mongodb.com/learn/learning-path/building-genai-apps-learning-badge-path

## Introduction to AI and Vector Search
Learn about the foundations of AI and how Atlas Vector Search fits in.

## Using Vector Search for Semantic Search
Learn all avout Atlas Vector Search as you build a semantic search feature.  Leverage both Atlas Search and Atlas Vector Search to identify the most relevant search results.

## Using Atlas Vector Search For RAG Applications
Learn how to implement retreval-augmented generation (RAG) with MongoDB in your application.  Learn waht retrieval augmented generation is and set it up uisng the MongoDB Python driver.

## Managing Atlas Vector Search Indexes
Learn how to manage your Atlas Vector Search indexes using the Atlas CLI and MongoDB Shell

## Data Ingestion for RAG Applications
Learn about the Data Ingestion Pipeline for retrieval-augmented generation (RAG) applications.

## Glossary
This glossary is Primarily taken from https://learn.mongodb.com/learn/course/mongodb-and-genai-glossary.  But may include some additions, or links to outside resources.

### Approximate Nearest Neighbor search (aNN)
Approximate Nearest Neighbor (aNN) search is a computational technique used to quickly find points in a dataset that are close to a given query point.  aNN doesn't search every point in the graph, it uses the HNSW (Hierarchical Navigable Small World) graph to search until it finds an approximate neighbor.

### Artifical Intelligence (AI)
A field in computer sciene that trains computers to simulate human intelligence.

### Atlas Vector Search
Atlas Vector Search is a featuer in MongoDB Atlas that allows users to perform fast and efficient similarity searches on high-dimensional data by using vector representations.  It enables semantic search cababilities by comparing vectors to find the most relevant or similar items to a given query.

see also; https://www.pinecone.io/learn/vector-similarity/
|Similarity Metric      | Vector Properties considered |
|-----------------------|------------------------------|
| Cosine similiarity    | Only direction               |
| Dot Product similarity| Magnitudes and direction     |
| Euclidean distance    | Magnitudes and directions    |
#### Cosine Similarity
[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) is a metric that uses the angle between two vectors to determien the similarity between those vectors.

Cosine does not take magnitude into account, and you can't use zero magnitude vectors with cosine. To measure cosine similarity, it is recommended to normalize the vectors and use dotProduct.

#### DotProdcut similarity
Dot Product similarity is a measure of similarity between two vectors in a vector space, calculated as the sum of the prodcuts of their corresponding components.

To use dotProduct, the vector must be normalized to unit length at index- and query-time.

#### Euclidean Distance
Euclidean similarity uses the distance between two vectors in a multidimensional space to calculatet the similarity between those vectors.  The underlying formula is derived from the Pythagorean theorem and generalized for any number of dimensions.

### Chunking
The process of breaking down large amounts of information or data into smaller, more manageable pieces, or "chunks".  These chunks are designed to be easily processed, stored, and retrieved by the AI model.

#### Chunk size
The maximum number of tokens contained in each chunk.

#### Chunk overlap
The number of overlapping tokens between two adjacent chunks.  This overlap will create duplicate data across chunks. Overlap can help preserve context between chunks and improve your result.

#### Chunking Strategy
see also: https://www.pinecone.io/learn/chunking-strategies/
Chunking strategy involves planning and implementing the optimal way to divide information into chunks to maximize the performance of the AI system.  Chumking strategy takes inot account the nature of the data, the specific requiremetns of the task, and the capabilities of the AI model.  The goal is to create chunks that are coherent, contextually relevant, and of and appropriate size to be processed effetively by the model.

### Context
#### Context window
The context window is the amount of data that an LLM can consider at one time.  The size of an LLM's context window is measured in tokens.

#### Context Recall
Context Recall represents how aligned the context retrieved is to the ground truth.

#### Context Precision 
Context Precision represents the ability of a retriever to order retrieved itesm by relevance to the ground truth.

#### Faithfulness
A measure of generation performance intended to quantify the factual correctness of a generated answer compared to the retrieved context.  I.e. how close is the answe to the chunks provided by the retriever.

### Dense Vectors
A dense vector is a numeric representation of data where most or all the dimensions contain non-zero values.  Dense vectors are commonly used in semantic searches and machine learning applications to represent complex, abstract features, such as the meaning of words, phrases, or entire documents, in  a continuous, high-dimensional space.

### Dimensions
The dimensions of a vector embedding represent the features or attributes of the data item in a high-dimensional space.

Each dimension corresponds to a particular aspect or characteristic of the data, and teh value in each dimension represents the strength or presence of tht charactieristics.

### Embeddings
An array of numerical data, or vectors, representing a piece of information, such as text, images, audio, video, etc.

#### Embedding Model
A deep learning model used to transform data into vectors, also known as embeddings.

### Ground Truth
A factually corrrect statement provided in a dataset to compare with LLM responses.

### Hierachical Navigable Small World (HNSW) graphs
[HNSW graphs](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) plot multiple point with long and short links, similar to [Navigable Small World (NSW) graph](https://en.wikipedia.org/wiki/Small-world_network). But unlike NSW graphs wihich are on one plane, HNSW graphs are made up of multiple layers, like a [skip list](https://en.wikipedia.org/wiki/Skip_list)

### Knowledge Base
In the context of RAG, a knowledge base is a repository that stores and organzies data, so it's easy to retrieve information that models use to generate responses.

### Model
A model is a mathematical representations or algorithm that is trained on data to recognize patterns and make predictions or decisions.  the algorithm dictates how the learning process happens, and the outcome of this learning process is a model.

#### Decoder Models
Decoder models use only the decoder of a Transformer model.  They take embeddings and generate an output based on the context provided by the embeddings.  Pretraining of these models usually revolves around predicting the next word in a sentence.

#### Encoder Models
Encoder models only use the encoder from the Transformer model.  Encoders take an input and return embeddings.



### Reciprocal Rank Fusion
Reciprocal Rank Fusion (RRF) is an information retrieval technique used to combine the ranked results from multiple search engines or algorithsm into a singl, more accurate ranking.  It works by giving higher importance to results that are ranked highly by any of the individual ranking systesm while ensuring that results ranked lower also get some consideration.
see also: https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a
