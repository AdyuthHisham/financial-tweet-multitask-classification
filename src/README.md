# src

- `TweetNormalizer.py`: tweet normalisation (mentions to `@USER`, links to `HTTPURL`, emoji to text). It follows the normaliser published with BERTweet
  (Nguyen et al., 2020); check upstream before reuse.
- `class_mapping.json`: the 21 company labels (20 tickers plus `OTHERS`) used by Approach B.
