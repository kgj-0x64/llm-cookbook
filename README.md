# Cookbook

## Embeddings

### 1. Visualization

1. **Visualizing embeddings in 2D/3D:** In the 3D plot, it starts appearing to eyes that very few negative (1/2 starred) reviews are clearly separable from positive (4/5 starred) reviews but many others are embedded in tight neighbouring or shared spaces. You wish we could do the visualization exercise in one more dimension and maybe the separation boxes/planes/lines would be clearer for more data points.

### 2. Classification

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
