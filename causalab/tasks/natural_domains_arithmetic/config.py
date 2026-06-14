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

# Few-shot prefixes for FR/ES presets (2 worked Q/A examples each).
_FS_WEEKDAYS_FR = (
    "Q: Quel jour est deux jours après lundi?\nA: mercredi\n"
    "Q: Quel jour est trois jours après vendredi?\nA: lundi\n"
)
_FS_WEEKDAYS_ES = (
    "Q: ¿Qué día de la semana es dos días después de lunes?\nA: miércoles\n"
    "Q: ¿Qué día de la semana es tres días después de viernes?\nA: lunes\n"
)
_FS_MONTHS_FR = (
    "Q: Quel mois est deux mois après janvier?\nA: mars\n"
    "Q: Quel mois est cinq mois après octobre?\nA: mars\n"
)
_FS_MONTHS_ES = (
    "Q: ¿Qué mes es dos meses después de enero?\nA: marzo\n"
    "Q: ¿Qué mes es cinco meses después de octubre?\nA: marzo\n"
)

# CJK and Hindi extensions. Offsets use Arabic numerals (1–7) to avoid
# grammatical ambiguity (Chinese 两 vs 二) and CJK/Devanagari tokenization complexity.
_DAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
_MONTHS_ZH = [
    "一月", "二月", "三月", "四月", "五月", "六月",
    "七月", "八月", "九月", "十月", "十一月", "十二月",
]
_DAYS_JA = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
_MONTHS_JA = [
    "一月", "二月", "三月", "四月", "五月", "六月",
    "七月", "八月", "九月", "十月", "十一月", "十二月",
]
_DAYS_HI = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
_MONTHS_HI = [
    "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]
# Korean (Hangul) weekdays. Like CJK, day names are single contiguous words and
# offsets use Arabic numerals (1–7) to avoid native-numeral tokenization issues.
_DAYS_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

# Shared Arabic-numeral offset list for ZH/JA/HI/KO (1–7).
_NUMBERS_ARABIC_7 = ["1", "2", "3", "4", "5", "6", "7"]
_NUMBER_TO_INT_ARABIC_7 = {str(i): i for i in range(1, 8)}

_FS_WEEKDAYS_ZH = (
    "Q: 星期一后2天是哪天？\nA: 星期三\n"
    "Q: 星期五后3天是哪天？\nA: 星期一\n"
)
_FS_MONTHS_ZH = (
    "Q: 一月后2个月是哪个月？\nA: 三月\n"
    "Q: 十月后5个月是哪个月？\nA: 三月\n"
)
_FS_WEEKDAYS_JA = (
    "Q: 月曜日の2日後は何曜日ですか？\nA: 水曜日\n"
    "Q: 金曜日の3日後は何曜日ですか？\nA: 月曜日\n"
)
_FS_MONTHS_JA = (
    "Q: 一月の2ヶ月後は何月ですか？\nA: 三月\n"
    "Q: 十月の5ヶ月後は何月ですか？\nA: 三月\n"
)
_FS_WEEKDAYS_HI = (
    "Q: सोमवार के 2 दिन बाद कौन सा दिन होगा?\nA: बुधवार\n"
    "Q: शुक्रवार के 3 दिन बाद कौन सा दिन होगा?\nA: सोमवार\n"
)
_FS_MONTHS_HI = (
    "Q: जनवरी के 2 महीने बाद कौन सा महीना होगा?\nA: मार्च\n"
    "Q: अक्टूबर के 5 महीने बाद कौन सा महीना होगा?\nA: मार्च\n"
)
_FS_WEEKDAYS_KO = (
    "Q: 월요일에서 2일 후는 무슨 요일입니까?\nA: 수요일\n"
    "Q: 금요일에서 3일 후는 무슨 요일입니까?\nA: 월요일\n"
)

# Low-resource Latin-script extensions (session 2026-06-14--low-resource-latin).
# Four families: Vietnamese (Austroasiatic), Swahili (Bantu), Turkish (Turkic),
# Indonesian (Austronesian). Latin script => output metrics valid on gemma3.
# Offsets use Arabic numerals 1-7 (shared _NUMBERS_ARABIC_7) to isolate the
# weekday concept from number-word tokenization, matching the zh/ja/hi/ko presets.
#
# NOTE (vi): Vietnamese day names are lexically ordinal numbers ("thu Hai" = 2nd
# day) and 6/7 share the "thu" prefix; only "Chu Nhat" (Sunday) differs. The
# startswith gate (multi-token) disambiguates them. The ordinal-numeric naming is
# a known interpretation caveat -- the weekday subspace may entangle with the
# number subspace (cross-checked via a number-arithmetic control this session).
_DAYS_VI = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ Nhật"]
_DAYS_SW = ["Jumatatu", "Jumanne", "Jumatano", "Alhamisi", "Ijumaa", "Jumamosi", "Jumapili"]
_DAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
_DAYS_ID = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

_FS_WEEKDAYS_VI = (
    "Q: 2 ngày sau thứ Hai là thứ mấy?\nA: thứ Tư\n"
    "Q: 3 ngày sau thứ Sáu là thứ mấy?\nA: thứ Hai\n"
)
_FS_WEEKDAYS_SW = (
    "Q: Siku 2 baada ya Jumatatu ni siku gani?\nA: Jumatano\n"
    "Q: Siku 3 baada ya Ijumaa ni siku gani?\nA: Jumatatu\n"
)
_FS_WEEKDAYS_TR = (
    "Q: Pazartesi gününden 2 gün sonra hangi gündür?\nA: Çarşamba\n"
    "Q: Cuma gününden 3 gün sonra hangi gündür?\nA: Pazartesi\n"
)
_FS_WEEKDAYS_ID = (
    "Q: Hari apa 2 hari setelah Senin?\nA: Rabu\n"
    "Q: Hari apa 3 hari setelah Jumat?\nA: Senin\n"
)

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
        template=_FS_WEEKDAYS_FR + "Q: Quel jour est {number} jours après {entity}?\nA:",
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
        template=_FS_WEEKDAYS_ES + "Q: ¿Cuál día de la semana está {number} días después del {entity}?\nA:",
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
        template=_FS_MONTHS_FR + "Q: Quel mois est {number} mois après {entity}?\nA:",
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
        template=_FS_MONTHS_ES + "Q: ¿Qué mes es {number} meses después de {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_zh": dict(
        entities=_DAYS_ZH,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_ZH + "Q: {entity}后{number}天是哪天？\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "months_zh": dict(
        entities=_MONTHS_ZH,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template=_FS_MONTHS_ZH + "Q: {entity}后{number}个月是哪个月？\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_ja": dict(
        entities=_DAYS_JA,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_JA + "Q: {entity}の{number}日後は何曜日ですか？\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "months_ja": dict(
        entities=_MONTHS_JA,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template=_FS_MONTHS_JA + "Q: {entity}の{number}ヶ月後は何月ですか？\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_hi": dict(
        entities=_DAYS_HI,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_HI + "Q: {entity} के {number} दिन बाद कौन सा दिन होगा?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_ko": dict(
        entities=_DAYS_KO,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_KO + "Q: {entity}에서 {number}일 후는 무슨 요일입니까?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_vi": dict(
        entities=_DAYS_VI,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_VI + "Q: {number} ngày sau {entity} là thứ mấy?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_sw": dict(
        entities=_DAYS_SW,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_SW + "Q: Siku {number} baada ya {entity} ni siku gani?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_tr": dict(
        entities=_DAYS_TR,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_TR + "Q: {entity} gününden {number} gün sonra hangi gündür?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "weekdays_id": dict(
        entities=_DAYS_ID,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=_FS_WEEKDAYS_ID + "Q: Hari apa {number} hari setelah {entity}?\nA:",
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "months_hi": dict(
        entities=_MONTHS_HI,
        numbers=_NUMBERS_ARABIC_7,
        number_to_int=_NUMBER_TO_INT_ARABIC_7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template=_FS_MONTHS_HI + "Q: {entity} के {number} महीने बाद कौन सा महीना होगा?\nA:",
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
    # Few-shot variants of the failing cycles. Each prefixes the test prompt with
    # 2 worked examples to provide in-context demonstrations of the modular
    # arithmetic pattern. Same entities and arithmetic, only the template differs.
    "moon_phases_fs": dict(
        entities=_MOON_QUARTERS,
        number_range=4,
        cyclic=True,
        modulus=4,
        number_is_cyclic=True,
        template=(
            "Q: What lunar phase is one phase after New Moon?\n"
            "A: First Quarter\n"
            "Q: What lunar phase is three phases after First Quarter?\n"
            "A: New Moon\n"
            "Q: What lunar phase is {number} phases after {entity}?\n"
            "A:"
        ),
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "solfege_fs": dict(
        entities=_SOLFEGE,
        number_range=7,
        cyclic=True,
        modulus=7,
        number_is_cyclic=True,
        template=(
            "Q: In solfège, what syllable is one step after Do?\n"
            "A: Re\n"
            "Q: In solfège, what syllable is three steps after Sol?\n"
            "A: Do\n"
            "Q: In solfège, what syllable is {number} steps after {entity}?\n"
            "A:"
        ),
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "compass_fs": dict(
        entities=_COMPASS_CARDINAL,
        number_range=4,
        cyclic=True,
        modulus=4,
        number_is_cyclic=True,
        template=(
            "Q: What direction is one turn clockwise from North?\n"
            "A: East\n"
            "Q: What direction is three turns clockwise from South?\n"
            "A: East\n"
            "Q: What direction is {number} turns clockwise from {entity}?\n"
            "A:"
        ),
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "zodiac_fs": dict(
        entities=_ZODIAC,
        number_range=7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template=(
            "Q: What zodiac sign is one sign after Aries?\n"
            "A: Taurus\n"
            "Q: What zodiac sign is three signs after Gemini?\n"
            "A: Virgo\n"
            "Q: What zodiac sign is {number} signs after {entity}?\n"
            "A:"
        ),
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
    "chinese_zodiac_fs": dict(
        entities=_CHINESE_ZODIAC,
        number_range=7,
        cyclic=True,
        modulus=12,
        number_is_cyclic=False,
        template=(
            "Q: In the Chinese zodiac, what animal comes one year after Rat?\n"
            "A: Ox\n"
            "Q: In the Chinese zodiac, what animal comes two years after Tiger?\n"
            "A: Dragon\n"
            "Q: In the Chinese zodiac, what animal comes {number} years after {entity}?\n"
            "A:"
        ),
        output_prefix=" ",
        result_entities=None,
        compute_result=None,
        entity_embedding=None,
    ),
}
