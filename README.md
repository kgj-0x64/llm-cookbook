# Cookbook

## Embeddings

### Text Classification

1. **Zero Shot Classification:** Turned into a fun puzzle experience! I was fun trying to improve the accuracy of the "prompt" (that is, sentences representing positive and negative classifications which then goes into the embedding model) versus what's given in the OpenAI blog, but in the end overall accuracy remained the same with higher precision but lower recall. It should be noted that the data is too skewed towards positive (particulary 5-star) reviews and their scores don't necessarily correlate with the opinion on product directly e.g. negative reviews could be complementing the product while saying that it was not what they had anticipated or that they don't like anything among this kind (category) of products.
