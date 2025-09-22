from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import random
import re

class BigramModel:
    def __init__(self, corpus, frequency_threshold=1):
        self.corpus = corpus
        self.frequency_threshold = frequency_threshold
        self.vocab, self.bigram_probs = self.analyze_bigrams(" ".join(corpus))

    def simple_tokenizer(self, text):
        """Simple tokenizer that splits text into words."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not self.frequency_threshold:
            return tokens

        # Count word frequencies
        word_counts = Counter(tokens)

        # Filter out infrequent words
        filtered_tokens = [
            token for token in tokens if word_counts[token] >= self.frequency_threshold
        ]
        return filtered_tokens

    def analyze_bigrams(self, text):
        """Analyze text to compute bigram probabilities."""
        words = self.simple_tokenizer(text)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return list(unigram_counts.keys()), bigram_probs

    def generate_text(self, start_word, num_words=20):
        """Generate text based on bigram probabilities."""
        current_word = start_word.lower()

        # Fallback if start_word not in vocab
        if current_word not in self.bigram_probs:
            current_word = random.choice(list(self.bigram_probs.keys()))

        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:
                # fallback: random restart
                current_word = random.choice(list(self.bigram_probs.keys()))
                generated_words.append(current_word)
                continue

            next_word = random.choices(
                list(next_words.keys()), weights=next_words.values()
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)

    def print_bigram_probs_matrix_python(self, vocab):
        """Print bigram probabilities as a matrix."""
        print(f"{'':<15}", end="")
        for word in vocab:
            print(f"{word:<15}", end="")
        print("\n" + "-" * (15 * (len(vocab) + 1)))

        for word1 in vocab:
            print(f"{word1:<15}", end="")
            for word2 in vocab:
                prob = self.bigram_probs.get(word1, {}).get(word2, 0)
                print(f"{prob:<15.2f}", end="")
            print()