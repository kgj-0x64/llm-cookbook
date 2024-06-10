# Cookbook

## Embeddings

### Word Vector

1. [Word2Vec Tutorial - The Skip-Gram Model - By Chris McCormick](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
2. [Word2Vec Tutorial Part 2 - Negative Sampling - By Chris McCormick](https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
3. [Coding Word2vec from Scratch](https://jaketae.github.io/study/word2vec/)

### Summary

| Aspect           | Tokenization                                   | Embeddings                                      |
|------------------|------------------------------------------------|-------------------------------------------------|
| Definition       | The process of splitting text into smaller units (tokens) such as words, subwords, or characters. | The representation of tokens as dense vectors in a continuous vector space. |
| Purpose          | To convert text into a format that can be processed by models. | To capture the semantic meaning and relationships between tokens. |
| Output           | A sequence of tokens (strings or subunits).    | A sequence of vectors (numerical representations). |
| Role in NLP      | Initial step in text processing.               | Used after tokenization for feature extraction and input to models. |
| Examples         | Splitting "Hello world!" into ["Hello", "world", "!"]. | Representing "Hello" as [0.25, -0.14, 0.44, ...]. |
| Techniques       | Word-level, subword-level (BPE, WordPiece), character-level. | Word2Vec, GloVe, FastText, BERT embeddings. |
| Data Dependency  | Does not inherently learn from data; it's a rule-based or deterministic process. | Learned from a large corpora to capture contextual information. |
| Flexibility      | Less flexible, depends on predefined rules.    | Highly flexible, adapts to various contexts and uses. |
| Computational Complexity | Low complexity, simple to implement.     | Higher complexity, requires training on large datasets. |
| Visualization Tool | https://www.pinecone.io/learn/tokenization/ | https://docs.wandb.ai/guides/app/features/panels/query-panel/embedding-projector |

### Visualization

1. **Visualizing embeddings in 2D/3D:** In the 3D plot, it starts appearing to eyes that very few negative (1/2 starred) reviews are clearly separable from positive (4/5 starred) reviews but many others are embedded in tight neighbouring or shared spaces. You wish we could do the visualization exercise in one more dimension and maybe the separation boxes/planes/lines would be clearer for more data points.

    - W&B Embedding Projector tool: E.g. [W&B Embedding Projector to visualize OpenAI Embeddings alongside the Amazon food reviews data that produced them](https://wandb.ai/_scott/openai_embeddings/reports/OpenAI-Embeddings-Table--VmlldzozNDYxNjkx)

2. **Chunking:** Every tokenizer or GPT model has a token limit (i.e. a maximum input length), so chunking helps in breaking text to process larger texts that exceed this limit, and it also allows concurrent processing potentially speeding up the overall processing time. E.g. [to translate a book](https://github.com/openai/openai-cookbook/blob/main/examples/book_translation/translate_latex_book.ipynb), we will first split the book into chunks, each roughly a page long, then translate each chunk, and finally stitch them back together.

3. **Custom or fine-tuned embeddings:** [OpenAI's embedding model weights cannot be fine-tuned, but we can use training data to customize embeddings to our application.](https://cookbook.openai.com/articles/text_comparison_examples)

### Classification

1. **Zero Shot Classification:** Turned into a fun puzzle experience! I was fun trying to improve the accuracy of the "prompt" (that is, sentences representing positive and negative classifications which then goes into the embedding model) versus what's given in the OpenAI blog, but in the end overall accuracy remained the same with higher precision but lower recall. It should be noted that the data is too skewed towards positive (particulary 5-star) reviews and their scores don't necessarily correlate with the opinion on product directly e.g. negative reviews could be complementing the product while saying that it was not what they had anticipated or that they don't like anything among this kind (category) of products.

2. **Multi-Class Classification:** Data has mostly unique users interacting with unique products. Representing a user as an average (or weighted average) of products (rather product embeddings) they have interacted with in the training set concludes in a weak correlation of similarity between that user embedding and this new product's embedding with their rating score on this new product. Tried extending the same experiment as a multi-class classification problem to understand if collaborative filtering or a neural network could be accurate at rating score classification, but both are doing worst. For this data, it can be inferred that review texts (rather their embeddings) show very reliable correlation with positive and negative sentiment classes but very poor correlation with 5 types of rating scores.

## Working With Foundation Model Services

### Rate Limiting

[It's recommended to use the 'tenacity' package or another exponential backoff implementation to better manage API rate limits, as hitting the API too much too fast can trigger rate limits. Using the following function ensures you get your embeddings as fast as possible.](https://github.com/openai/openai-cookbook/blob/main/examples/Using_embeddings.ipynb)

```python
# Negative example (slow and rate-limited)
from openai import OpenAI
client = OpenAI()

num_embeddings = 10000 # Some large number
for i in range(num_embeddings):
    embedding = client.embeddings.create(
        input="Your text goes here", model="text-embedding-3-small"
    ).data[0].embedding
    print(len(embedding))
```

```python
# Best practice
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI
client = OpenAI()

# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding

embedding = get_embedding("Your text goes here", model="text-embedding-3-small")
print(len(embedding))
```

[However, if you're processing large volumes of batch data, where throughput matters more than latency, there are a few other things you can do in addition to backoff and retry.](https://cookbook.openai.com/examples/how_to_handle_rate_limits) e.g. batching requests pand roactively adding delay between requests. Here is an example script for parallel processing large quantities of API requests: [api_request_parallel_processor.py.](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)
