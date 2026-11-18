"""Generate the committed, synthetic multilingual demo corpus from authored word lists.

The templates illustrate repeated subword patterns, not a natural-language benchmark.
No source text is downloaded or copied from an external corpus.
"""

import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Language:
    """Define a small synthetic grammar and lexical inventory."""

    name: str
    subjects: tuple[str, ...]
    verbs: tuple[str, ...]
    objects: tuple[str, ...]
    endings: tuple[str, ...]
    template: str


LANGUAGES = (
    Language(
        "English",
        ("the student", "the teacher", "the writer", "the friend"),
        ("reads", "writes", "reviews", "understands"),
        (
            "the book",
            "the message",
            "the question",
            "the answer",
            "the story",
            "the lesson",
            "the language",
            "the text",
        ),
        ("today", "at school", "at home", "in the morning"),
        "{subject} {verb} {object} {ending} {number}",
    ),
    Language(
        "Thai",
        ("เด็ก", "ครู", "นักเรียน", "เพื่อน"),
        ("อ่าน", "เขียน", "ดู", "เรียน"),
        ("หนังสือ", "ข้อความ", "ภาษา", "เรื่องราว", "จดหมาย", "คำถาม", "คำตอบ", "บทเรียน"),
        ("วันนี้", "ที่โรงเรียน", "ที่บ้าน", "ตอนเช้า"),
        "{subject}{verb}{object}{ending} {number}",
    ),
    Language(
        "Chinese",
        ("学生", "老师", "作家", "朋友"),
        ("阅读", "书写", "复习", "理解"),
        ("书本", "消息", "问题", "答案", "故事", "课程", "语言", "文字"),
        ("今天", "在学校", "在家里", "在早上"),
        "{ending}{subject}{verb}{object} {number}",
    ),
    Language(
        "Japanese",
        ("学生は", "先生は", "作家は", "友達は"),
        ("読む", "書く", "復習する", "理解する"),
        ("本", "メッセージ", "質問", "答え", "物語", "授業", "言語", "文章"),
        ("今日", "学校で", "家で", "朝に"),
        "{ending}{subject}{object}を{verb} {number}",
    ),
    Language(
        "Korean",
        ("학생은", "선생님은", "작가는", "친구는"),
        ("읽는다", "쓴다", "복습한다", "이해한다"),
        ("책", "메시지", "질문", "답변", "이야기", "수업", "언어", "문장"),
        ("오늘", "학교에서", "집에서", "아침에"),
        "{subject} {ending} {object} {verb} {number}",
    ),
    Language(
        "Arabic",
        ("الطالب", "المعلم", "الكاتب", "الصديق"),
        ("يقرأ", "يكتب", "يراجع", "يفهم"),
        ("الكتاب", "الرسالة", "السؤال", "الجواب", "القصة", "الدرس", "النص", "اللغة"),
        ("اليوم", "في المدرسة", "في المنزل", "في الصباح"),
        "{subject} {verb} {object} {ending} {number}",
    ),
    Language(
        "Hindi",
        ("विद्यार्थी", "शिक्षक", "लेखक", "मित्र"),
        ("पढ़ता है", "लिखता है", "देखता है", "सीखता है"),
        ("किताब", "संदेश", "प्रश्न", "उत्तर", "कहानी", "पाठ", "भाषा", "लेख"),
        ("आज", "सुबह", "घर पर", "विद्यालय में"),
        "{subject} {ending} {object} {verb} {number}",
    ),
    Language(
        "French",
        ("l'élève", "le professeur", "l'auteur", "l'ami"),
        ("lit", "écrit", "révise", "comprend"),
        (
            "le livre",
            "le message",
            "la question",
            "la réponse",
            "l'histoire",
            "la leçon",
            "la langue",
            "le texte",
        ),
        ("aujourd'hui", "à l'école", "à la maison", "le matin"),
        "{subject} {verb} {object} {ending} {number}",
    ),
)


def generate() -> list[Path]:
    """Create disjoint training and held-out sentences from the same fixed grammars."""
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261118)
    training: list[str] = []
    tests: list[Path] = []
    for language in LANGUAGES:
        keys = rng.sample(range(4 * 4 * 8 * 4 * 100), 2400)
        lines: list[str] = []
        for key in keys:
            number = key % 100
            key //= 100
            ending = key % 4
            key //= 4
            obj = key % 8
            key //= 8
            verb = key % 4
            subject = key // 4
            lines.append(
                language.template.format(
                    subject=language.subjects[subject],
                    verb=language.verbs[verb],
                    object=language.objects[obj],
                    ending=language.endings[ending],
                    number=number,
                )
            )
        training.extend(lines[:2000])
        assert set(lines[:2000]).isdisjoint(lines[2000:])
        path = ROOT / f"heldout-{language.name.lower()}.txt"
        path.write_text("\n".join(lines[2000:]) + "\n", encoding="utf-8")
        tests.append(path)
    rng.shuffle(training)
    (ROOT / "train.txt").write_text("\n".join(training) + "\n", encoding="utf-8")
    return tests


if __name__ == "__main__":
    generate()
