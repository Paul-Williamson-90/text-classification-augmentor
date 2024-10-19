import re
import numpy as np


class TextNoiseAugmentor:
    @staticmethod
    def _text_splitter(
        text: str, split_chars: list[str] = [" ", ". ", ", ", "! ", "? ", "\n"]
    ) -> tuple[list[str], list[str]]:
        pattern = "|".join(map(re.escape, split_chars))
        split_pattern = re.compile(pattern)
        matches = list(split_pattern.finditer(text))
        text_split = split_pattern.split(text)
        delimiters = [match.group(0) for match in matches]
        return text_split, delimiters

    @staticmethod
    def _sub_text_mutator(text_split: list[str], rate: float = 0.2) -> list[str]:
        final_split_text = []
        for part in text_split:
            if np.random.rand() < rate and len(part) > 3:
                length = len(part)
                start_index = np.random.randint(0, length - 1)
                choice = np.random.choice(["skip", "switch", "insert", "replace"])
                if choice == "skip":
                    warped = part[:start_index] + part[start_index + 1 :]
                elif choice == "switch":
                    warped = (
                        part[:start_index]
                        + part[start_index + 1]
                        + part[start_index]
                        + part[start_index + 2 :]
                    )
                elif choice == "replace":
                    alphabet = "abcdefghijklmnopqrstuvwxyz"
                    replace = np.random.choice(list(alphabet))
                    warped = part[:start_index] + replace + part[start_index + 1 :]
                else:
                    alphabet = "abcdefghijklmnopqrstuvwxyz"
                    insert = np.random.choice(list(alphabet))
                    warped = part[:start_index] + insert + part[start_index:]
                final_split_text.append(warped)
            else:
                final_split_text.append(part)
        return final_split_text

    @staticmethod
    def _reconstruct_text(text_split: list[str], delimiters: list[str]) -> str:
        reconstructed_text = "".join(
            [
                part + (delimiters[i] if i < len(delimiters) else "")
                for i, part in enumerate(text_split)
            ]
        )
        return reconstructed_text

    @staticmethod
    def augment_text(text: str, rate: float = 0.2) -> str:
        text_split, delimiters = TextNoiseAugmentor._text_splitter(text)
        mutated_text_split = TextNoiseAugmentor._sub_text_mutator(text_split, rate)
        final_text = TextNoiseAugmentor._reconstruct_text(mutated_text_split, delimiters)
        return final_text
