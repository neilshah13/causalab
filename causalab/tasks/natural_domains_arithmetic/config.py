"""Configuration for the natural_domains_arithmetic factory task.

Unified config for weekdays, months, and hours domains. Each shares
the same causal DAG: (entity, number) → result → raw_output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class NaturalDomainConfig:
    """Configuration for a natural-domain arithmetic task.

    Attributes:
        domain_type: One of "weekdays", "months", "hours", "integer", "alphabet", "age".
        entities: The domain values (days or months).
        numbers: Word-form numbers ("one", "two", ...).
        number_to_int: Mapping from word to integer.
        cyclic: Whether entities wrap around.
        modulus: Wrap-around modulus for cyclic domains.
        number_is_cyclic: Whether the number variable is also cyclic.
        template: Prompt template with {entity} and {number} placeholders.
        output_prefix: String prepended to result in raw_output.
        result_entities: Output domain if different from entities.
        compute_result: Custom result function; None uses default cyclic arithmetic.
        entity_embedding: Custom embedding function for entities.
        seed: Random seed.
    """

    domain_type: str
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    number_to_int: dict[str, int] = field(default_factory=dict)
    cyclic: bool = True
    modulus: int | None = None
    number_is_cyclic: bool = False
    template: str | list[str] = ""
    output_prefix: str = " "
    result_entities: list[str] | None = None
    compute_result: Callable[[str, str, "NaturalDomainConfig"], str] | None = None
    entity_embedding: Callable[[str], list[float]] | None = None
    number_range: int | None = None
    number_groups: list[list[int]] | None = None  # e.g. [[1,7],[8,14],[15,21]]
    seed: int = 42

    def __post_init__(self) -> None:
        valid = set(DOMAIN_PRESETS.keys())
        if self.domain_type not in valid:
            raise ValueError(
                f"domain_type must be one of {sorted(valid)}, got '{self.domain_type}'"
            )
        # Auto-fill from preset when entities list is empty. Skip overriding
        # fields the user set explicitly so YAML/runner overrides win over
        # the preset default (e.g. when running an old artifact that needs
        # a different number_range or result_entities than the current preset).
        if not self.entities:
            preset = DOMAIN_PRESETS[self.domain_type]
            explicit_skip = {
                k
                for k in ("number_range", "result_entities")
                if getattr(self, k) is not None
            }
            for k, v in preset.items():
                if k in explicit_skip:
                    continue
                setattr(self, k, v)
        # Materialize numbers/number_to_int from number_range. Presets that set
        # ``numbers`` directly (e.g. age uses digit strings) skip this.
        if self.number_range is not None and not self.numbers:
            if self.number_range > len(_ALL_NUMBER_WORDS):
                raise ValueError(
                    f"number_range={self.number_range} exceeds available "
                    f"number words ({len(_ALL_NUMBER_WORDS)})"
                )
            self.numbers = _ALL_NUMBER_WORDS[: self.number_range]
            self.number_to_int = {n: i + 1 for i, n in enumerate(self.numbers)}


# ---------------------------------------------------------------------------
# Preset data
# ---------------------------------------------------------------------------

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_HOURS = [str(h) for h in range(1, 25)]
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Multilingual extensions for the Stage-0 encoding gate. These mirror the
# English weekdays/months presets exactly in structure; only entity strings,
# number words, and the template body are translated.
_DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]
_MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
_NUMBERS_FR = ["un", "deux", "trois", "quatre", "cinq", "six", "sept"]
_NUMBERS_ES = ["uno", "dos", "tres", "cuatro", "cinco", "seis", "siete"]
_NUMBER_TO_INT_FR = {n: i + 1 for i, n in enumerate(_NUMBERS_FR)}
_NUMBER_TO_INT_ES = {n: i + 1 for i, n in enumerate(_NUMBERS_ES)}

# English-domain extensions for the Stage-0 encoding gate. Each list orders
# the entities cyclically; modular arithmetic uses index-mod-modulus.
# Multi-word entities are kept whole because raw_output uses startswith
# checking — the model emits the first token (max_new_tokens=1) and the
# expected raw_output begins with that token, so first-token disambiguation
# is what actually carries classification weight.
_MOON_QUARTERS = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]
_SOLFEGE = ["Do", "Re", "Mi", "Fa", "Sol", "La", "Ti"]
_COMPASS_CARDINAL = ["North", "East", "South", "West"]
_ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
_CHINESE_ZODIAC = [
    "Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake",
    "Horse", "Goat", "Monkey", "Rooster", "Dog", "Pig",
]
_ALL_NUMBER_WORDS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "twenty-one",
    "twenty-two",
    "twenty-three",
    "twenty-four",
    "twenty-five",
    "twenty-six",
    "twenty-seven",
    "twenty-eight",
    "twenty-nine",
    "thirty",
    "thirty-one",
    "thirty-two",
    "thirty-three",
    "thirty-four",
    "thirty-five",
    "thirty-six",
    "thirty-seven",
    "thirty-eight",
    "thirty-nine",
    "forty",
    "forty-one",
    "forty-two",
    "forty-three",
    "forty-four",
    "forty-five",
    "forty-six",
    "forty-seven",
    "forty-eight",
]
_ALL_NUMBER_TO_INT = {name: i + 1 for i, name in enumerate(_ALL_NUMBER_WORDS)}

_INTEGER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "twenty-one",
    "twenty-two",
    "twenty-three",
    "twenty-four",
    "twenty-five",
    "twenty-six",
    "twenty-seven",
    "twenty-eight",
    "twenty-nine",
    "thirty",
    "thirty-one",
    "thirty-two",
    "thirty-three",
    "thirty-four",
    "thirty-five",
    "thirty-six",
    "thirty-seven",
    "thirty-eight",
    "thirty-nine",
    "forty",
    "forty-one",
    "forty-two",
    "forty-three",
    "forty-four",
    "forty-five",
    "forty-six",
    "forty-seven",
    "forty-eight",
    "forty-nine",
    "fifty",
]
_INTEGER_WORD_TO_INT: dict[str, int] = {
    name: i for i, name in enumerate(_INTEGER_WORDS)
}

DOMAIN_PRESETS: dict[str, dict] = {
    "weekdays": dict(
        entities=_DAYS,
        number_range=7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template="Q: What day is {number} days after {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "months": dict(
        entities=_MONTHS,
        number_range=7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template="Q: What month is {number} months after {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "hours": dict(
        entities=_HOURS,
        number_range=24,
        cyclic=True,
        modulus=24,
        number_is_cyclic=False,
        template="Q: What hour comes {number} hours after {entity} on a clock?\nA: ",
        output_prefix="",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "integer": dict(
        entities=_INTEGER_WORDS[1:16],  # "one" … "fifteen"
        number_range=9,  # "one" … "nine"
        cyclic=False,
        modulus=None,
        number_is_cyclic=False,
        template="Q: What is {number} added to {entity}?\nA:",
        output_prefix=" ",
        result_entities=[str(i) for i in range(2, 26)],  # 2 … 25
        compute_result=lambda entity, number, cfg, _w2i=_INTEGER_WORD_TO_INT: str(
            _w2i[entity] + cfg.number_to_int[number]
        ),
        # Embed as integer value (not list index) for both word-form entities
        # and digit-form result values.
        entity_embedding=lambda v, _w2i=_INTEGER_WORD_TO_INT: (
            [float(_w2i[v])] if v in _w2i else [float(v)]
        ),
    ),
    "age": dict(
        entities=[
            str(i) for i in range(1, 100)
        ],  # "1" … "99"; pairs with entity+number > 100 are filtered out
        numbers=[str(i) for i in range(1, 11)],  # "1" … "10" (digit form, not word)
        number_to_int={str(i): i for i in range(1, 11)},
        cyclic=False,
        modulus=None,
        number_is_cyclic=False,
        template=(
            "Alice is {entity} years old. "
            "Bob is {number} years older than Alice. "
            "Q: How old is Bob?\nA: Bob is "
        ),
        output_prefix="",
        result_entities=[str(i) for i in range(10, 101)],  # "10" … "100"
        compute_result=lambda entity, number, cfg: str(int(entity) + int(number)),
        entity_embedding=lambda v: [float(v)],
    ),
    "alphabet": dict(
        entities=_LETTERS[
            :25
        ],  # A-Y; (entity, number) pairs whose result exceeds Z are filtered out
        number_range=4,
        cyclic=False,
        modulus=None,
        number_is_cyclic=False,
        template=(
            "Consider letters in the alphabet. "
            "Starting at letter {entity}, we increment by {number}. The result is letter"
        ),
        output_prefix=" ",
        # result_entities[number_range:] gives letters with full coverage:
        # each result class is reachable by every increment 1..number_range from
        # some valid entity. Keep this in sync with number_range.
        result_entities=_LETTERS[4:],
        compute_result=lambda entity, number, cfg: chr(
            ord(entity) + cfg.number_to_int[number]
        ),
        entity_embedding=None,
    ),
    "weekdays_fr": dict(
        entities=_DAYS_FR,
        numbers=_NUMBERS_FR,
        number_to_int=_NUMBER_TO_INT_FR,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template="Q: Quel jour est {number} jours après {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_es": dict(
        entities=_DAYS_ES,
        numbers=_NUMBERS_ES,
        number_to_int=_NUMBER_TO_INT_ES,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template="Q: ¿Qué día es {number} días después de {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "months_fr": dict(
        entities=_MONTHS_FR,
        numbers=_NUMBERS_FR,
        number_to_int=_NUMBER_TO_INT_FR,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template="Q: Quel mois est {number} mois après {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "months_es": dict(
        entities=_MONTHS_ES,
        numbers=_NUMBERS_ES,
        number_to_int=_NUMBER_TO_INT_ES,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template="Q: ¿Qué mes es {number} meses después de {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "moon_phases": dict(
        entities=_MOON_QUARTERS,
        number_range=4,
        cyclic=True,
        modulus=4,
        number_is_cyclic=True,
        template="Q: What lunar phase is {number} phases after {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "solfege": dict(
        entities=_SOLFEGE,
        number_range=7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template="Q: In solfège, what syllable is {number} steps after {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "compass": dict(
        entities=_COMPASS_CARDINAL,
        number_range=4,
        cyclic=True,
        modulus=4,
        number_is_cyclic=True,
        template="Q: What direction is {number} turns clockwise from {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "zodiac": dict(
        entities=_ZODIAC,
        number_range=7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template="Q: What zodiac sign is {number} signs after {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "chinese_zodiac": dict(
        entities=_CHINESE_ZODIAC,
        number_range=7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template="Q: In the Chinese zodiac, what animal year is {number} years after the year of the {entity}?\nA: the year of the",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
}
