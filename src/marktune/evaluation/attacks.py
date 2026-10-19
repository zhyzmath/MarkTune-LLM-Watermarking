"""Main-body text-editing attacks and realized tokenizer edit counts."""

import random


def validate_ratio(ratio):
    if not 0 <= ratio <= 1:
        raise ValueError("Attack ratio must lie in [0, 1]")


def delete_words(text, ratio, rng):
    """Independent word deletion, with ratio equal to deletion probability."""
    validate_ratio(ratio)
    return " ".join(word for word in text.split() if rng.random() >= ratio)


def wordnet_candidates(word):
    from nltk.corpus import wordnet

    # Resources are installed explicitly; attacks never download corpora implicitly.
    return [
        [lemma.name().replace("_", " ") for lemma in synset.lemmas()[1:]]
        for synset in wordnet.synsets(word)
        if len(synset.lemmas()) > 1
    ]


def substitute_words(text, ratio, rng, candidates=wordnet_candidates):
    """Replace up to floor(ratio * word_count) positions with WordNet synonyms."""
    validate_ratio(ratio)
    words = text.split()
    synonyms = {word: candidates(word) for word in set(words)}
    eligible = [index for index, word in enumerate(words) if synonyms[word]]
    for index in rng.sample(eligible, min(int(ratio * len(words)), len(eligible))):
        chosen = rng.choice(synonyms[words[index]])
        words[index] = rng.choice(chosen) if isinstance(chosen, (list, tuple)) else chosen
    return " ".join(words)


def edit_counts(source, target):
    """Minimum Levenshtein counts; ties prefer substitution, deletion, insertion."""
    previous = [(j, 0, 0, j) for j in range(len(target) + 1)]
    # Tuples are distance, substitutions, deletions, insertions.
    for i, old in enumerate(source, 1):
        current = [(i, 0, i, 0)]
        for j, new in enumerate(target, 1):
            if old == new:
                current.append(previous[j - 1])
            else:
                cost, sub, delete, insert = previous[j - 1]
                substitution = (cost + 1, sub + 1, delete, insert)
                cost, sub, delete, insert = previous[j]
                deletion = (cost + 1, sub, delete + 1, insert)
                cost, sub, delete, insert = current[j - 1]
                insertion = (cost + 1, sub, delete, insert + 1)
                current.append(min([substitution, deletion, insertion], key=lambda row: row[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return dict(insertions=insertions, deletions=deletions, substitutions=substitutions, distance=distance)


class Dipper:
    def __init__(self, device, diversity, max_new_tokens=200):
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        if diversity not in (0, 20, 40, 60, 80, 100):
            raise ValueError("DIPPER diversity must be a multiple of 20 in [0, 100]")
        self.diversity, self.device, self.max_new_tokens = diversity, device, max_new_tokens
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.model = (
            T5ForConditionalGeneration.from_pretrained(
                "kalpeshk2011/dipper-paraphraser-xxl", torch_dtype=dtype, trust_remote_code=False
            )
            .to(device)
            .eval()
        )

    def __call__(self, text, prompt=""):
        import torch
        from nltk.tokenize import sent_tokenize

        prefix = " ".join(prompt.split())
        output = []
        for sentence in sent_tokenize(" ".join(text.split())):
            code = 100 - self.diversity
            value = f"lexical = {code}, order = {code} {prefix} <sent> {sentence} </sent>"
            encoded = self.tokenizer(value, return_tensors="pt", truncation=True).to(self.device)
            with torch.inference_mode():
                ids = self.model.generate(
                    **encoded, max_new_tokens=self.max_new_tokens, do_sample=True, top_p=0.75, top_k=0
                )
            fragment = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            output.append(fragment)
            prefix = f"{prefix} {fragment}".strip()
        return " ".join(output)


class RoundtripTranslation:
    def __init__(self, device):
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device
        self.models, self.tokenizers = [], []
        for direction in ("en-fr", "fr-en"):
            name = f"Helsinki-NLP/opus-mt-{direction}"
            self.tokenizers.append(MarianTokenizer.from_pretrained(name))
            self.models.append(MarianMTModel.from_pretrained(name, trust_remote_code=False).to(device).eval())

    def __call__(self, text, prompt=""):
        import torch

        for tokenizer, model in zip(self.tokenizers, self.models):
            encoded = tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            with torch.inference_mode():
                ids = model.generate(**encoded)
            text = tokenizer.decode(ids[0], skip_special_tokens=True)
        return text


def attack_records(records, tokenizer, attack, *, method, ratio=None, diversity=None):
    for row in records:
        source = tokenizer.encode(row["text"], add_special_tokens=False)
        text = attack(row["text"], row["prompt"])
        target = tokenizer.encode(text, add_special_tokens=False)
        result = {key: value for key, value in row.items() if key not in ("score", "perplexity", "token_ids")}
        result.update(
            text=text,
            token_ids=target,
            attack={
                "method": method,
                "ratio": ratio,
                "diversity": diversity,
                "source_tokens": len(source),
                "target_tokens": len(target),
                "edits": edit_counts(source, target),
            },
        )
        yield result


def run(args, config):
    from transformers import AutoTokenizer
    from .generation import seed_everything
    from .io import read_records, write_jsonl

    seed_everything(config["evaluation"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or config["model"]["name"],
        trust_remote_code=False,
        revision=config["model"].get("revision") if not args.tokenizer else None,
    )
    rng = random.Random(config["evaluation"]["seed"])
    if args.attack in ("delete", "substitute"):
        if args.ratio is None:
            raise ValueError("Deletion and substitution require --ratio")
        function = delete_words if args.attack == "delete" else substitute_words

        def attack(text, prompt):
            return function(text, args.ratio, rng)
    elif args.attack == "dipper":
        if args.diversity is None:
            raise ValueError("DIPPER requires --diversity (20 or 60 for the paper)")
        attack = Dipper(config["runtime"]["device"], args.diversity, config["evaluation"]["response_tokens"])
    else:
        attack = RoundtripTranslation(config["runtime"]["device"])
    write_jsonl(
        args.output,
        attack_records(
            read_records(args.input), tokenizer, attack, method=args.attack, ratio=args.ratio, diversity=args.diversity
        ),
    )
