"""Character checks shared by every eval script."""
import re
import unicodedata


def text_checks(text):
    normalized = unicodedata.normalize('NFKD', text).casefold()
    nonempty = bool(text.strip())
    control_ok = not any(unicodedata.category(c).startswith('C') and c not in '\n\r\t'
                         for c in text)
    control_ok = (control_ok and '�' not in text
                  and not bool(re.search(r'<\|[^>]*\|>', text)))
    return dict(literal_no_e=nonempty and 'e' not in text.casefold(),
                normalized_no_e=nonempty and 'e' not in normalized,
                control_ok=control_ok, e_count=text.count('e')+text.count('E'))


def e_pass(text):
    """Nonempty and no e of any kind, including accented forms."""
    return text_checks(text)['normalized_no_e']
