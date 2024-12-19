# Introduction to AI and Vector Search
This course starts with a introduction to AI and then goes over a brief history of models and ends with a lesson on Transformer models.

![ai.layered](../images/ai.drawio.svg)

## Glossary

### Artifical Intelligence
A field in computer science that trains computers to simulate human intelligence.

### Machine Learning
A field in AI that creates algorithms and models to help computers digest vast amounts of data, which can be structured or unstrutured.

### Deep Learning
Deep learning is as subset of machine learning that involves algorithsm inspired by the structure and functions of the brain, known as artificial neural networks.

### Generative AI
GenAI leverages large language models to create text, images, video, and audio based on a given prompt.

### Large Language Models (LLM)
A type of artifical intelligence (AI) that specializes in processing text data and generating human-like text.

These models can recognize complex patterns, understand context, make predictions, and more.

### Natural Language Processing
NLP gives computers teh ability to understand and work with human languages.

### Neural Networks
A computational model inspired  by the structure and functional aspects of the brain.

A neural network is used to model complex relationships between inputs and outputs and to find patterns in data.

#### Feed foward neural network
see also [wiki](https://en.wikipedia.org/wiki/Feedforward_neural_network)
Feedforward neural networks are the simplest type of neural network.  The connectiosn between nodes are linear and move forward.  Feedforward neural networks are widely used for tasks like classification.

![neural](../images/neural.drawio.svg)

#### Recurrent Neural Networks (RNNs)
see also [wiki](https://en.wikipedia.org/wiki/Recurrent_neural_network)

see also [backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

see also [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)

Similar to feedforward network, RNNS move forward through nodes, but RNNs have the ability to loop backwards.  This allows RNNS to not only leverage the dat it's been given, but it cona also use what it has learned from the previous input.

### Tranformer Model
see [wiki](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))

![transformer](../images/transformer.drawio.svg)

#### Encoder
Responsible for context.
Processes in paralell.
#### Decoder
Uses context provided by encoder to make predictions.

### Sequence to Sequence Models
A Sequence to Sequene model uses both encoders and decoders. Sequence to sequence models excel at task that involve taking an input and generating a sentence based on that input.
* summarization
* translation

Representative models
[BART](https://huggingface.co/transformers/model_doc/bart)
[mBART](https://huggingface.co/transformers/model_doc/mbart)
[Marian](https://huggingface.co/transformers/model_doc/marian)
[T5](https://huggingface.co/transformers/model_doc/t5)

### Encoder Models
Encoder models only use the encoder from the Transformer model.  Encoders take an input and return embeddings.
Encoders have **bi-directional attention** this allows them to use text that comes before and after the target section, making them particularly good at understanding context.

Best suited for tasks requiring an understanding of teh full sentence, such as sentence classification, named entity recognition and extractive question answering (extract answers from documents).

Representative models:
[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert)
[BERT](https://huggingface.co/docs/transformers/model_doc/bert)
[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra)
[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)

### Decoder Models
Decoder models use only the decoder of a Transformer model.  They take embeddings and generate an output based on the context provided by the embeddings.  Pretraining of these models usually revolves around predicting the next word in a sentence.

Decoders are **unidirectional** which means the prediction for each output token can only depend on tokens generated before it.

These models are best suited for tasks envolving text generation.

Representative models:

[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt)
[GPT-2](https://huggingface.co/transformers/model_doc/gpt2)
[CTRL](https://huggingface.co/transformers/model_doc/ctrl)
[Transformer XL](https://huggingface.co/transformers/model_doc/transfo-xl)
